//! The step record: everything that varies from step to step, in one buffer.
//!
//! Kernels do not take the rows of a step as arguments. The conductor writes
//! them into one page-locked buffer, each device gets that buffer with one
//! copy, and launches carry only pointers into it and sizes. So a step of any
//! shape costs the same number of copies and the same launches.
//!
//! The record starts with what the layer kernels of `ares-cuda` read
//! (`kernels/layer_step.cuh`, the layout of [`ares_cuda::StepRecord`]): a
//! header of counts and byte offsets, then the segments, the tokens, the
//! positions and the rows whose logits are wanted. A device pointer to the
//! record is therefore the `step` argument of those kernels. What only the
//! runtime and the models read follows, found through a second header.
//!
//! Layout, little-endian `u32` words, every array at a multiple of 16 bytes:
//!
//! | section | words | content |
//! | --- | --- | --- |
//! | kernel header | 8 | rows, segments, wanted rows, 0, byte offsets of segments, tokens, positions, wanted rows |
//! | runtime header | 16 | magic, step number, counts of runs, updates, entries and commits, byte offsets of the sections below, the longest segment, the longest commit |
//! | segments | 8 each | first row, rows, sequence slot, accepted rows, first position, flags (1: direct), 0, 0 |
//! | tokens | 1 per row | |
//! | positions | 4 per row | time, height, width, extra: `(p, p, p, 0)` for text |
//! | row extras | 2 per row | sequence slot, context |
//! | wanted rows | 1 each | the rows whose logits are wanted, in order |
//! | runs | 2 each | first row, number of rows: the wanted rows as ranges |
//! | sampler rows | 32 each | how each wanted row is picked ([`SampleRow`]), in the order of the wanted rows |
//! | corrections | 4 each | the logit corrections of the wanted rows ([`Entry`]); a sampler row names its own |
//! | masks | mask words each | the sets of tokens the wanted rows may pick, a bit per token; a sampler row names its own |
//! | commit record | 8 + 8 each | a kernel header and the segments of the last step whose rows were drafts, with the rows that were accepted |
//! | updates | 3 each | sequence slot, first page-table entry, number of entries |
//! | entries | 1 each | page-table entries, for the updates in order |
//! | state copies | 3 each | 0 save or 1 restore, sequence slot, checkpoint place |
//! | page copies | 3 each | page to copy from, page to copy to, cells to copy |
//!
//! A segment is **direct** when all of its rows are certain to be accepted (a
//! slice of a prompt, a generation row without drafts): the recurrent kernels
//! then write the sequence's state themselves. The rows of any other segment
//! leave recurrent state untouched, and the next step carries, in its commit
//! record, how many of them were accepted. The commit record is itself in the
//! kernels' layout, so its device address is the `step` argument of the commit
//! kernels.
//!
//! A window whose drafts the model makes itself (flag 2 of its segment) has
//! no tokens for its draft rows when the record is written: the model's draft
//! block picks them on the device, before the rows are embedded, and writes
//! them into the token array of the device's copy of the record. The host
//! reads them back with the picks.
//!
//! # The prefix cache
//!
//! A prompt that starts from cached pages carries three more things.
//!
//! - A **replay** segment (flag 4, always direct too): rows of a prompt whose
//!   cells are cached already. They rebuild the recurrent state: each row
//!   attends the cells of its own sequence, its own cell included, and appends
//!   nothing to the pages, which other sequences read. They have no logits
//!   wanted. A segment is replay or it is not; the conductor never puts
//!   replay rows and rows that append in one segment, nor two segments of one
//!   slot in one step.
//! - **State copies**: a *save* copies the recurrent state of a slot into a
//!   checkpoint place, a *restore* copies a place into the slot. A place is
//!   a number below the budget of checkpoints; the model's device part holds
//!   that many places.
//! - **Page copies**: the first `cells` cells of page `from` go into page
//!   `to`, for every part of the state that lives in pages. This is the copy
//!   on write of a shared, partly used page ([`ares_state::CopyOnWrite`]).
//!
//! **The order on a device** is: the commit of the step before (its record
//! is this one's commit record), the saves, the page copies, the restores, the
//! rows. The saves come after the commit because a save names the position the
//! committed rows reached; they come before everything that writes a slot, so
//! that a slot a finished request left may be taken by the next request in
//! the very record that saves the state it left; the restores come before
//! the rows that read the restored state. The conductor never restores a place
//! that a later save of the same record overwrites, and a restore may read
//! a place that an earlier save of the same record wrote.
//!
//! The *context* is a number the host derives from the sequence's tokens up to
//! and including the row's token. A real model has no use for it (attention
//! reads the sequence's pages); the synthetic model has no attention kernel,
//! so this is how a row's result comes to depend on its whole sequence.

use ares_cuda::{Segment, StepDims};
use ares_sampler::{ENTRIES_PER_ROW, Entry, SampleParams, SampleRow, TokenMask};

const MAGIC: u32 = 0x5354_4550;
/// Bytes of the kernels' header.
const KERNEL_HEADER: usize = 32;
/// Words of both headers.
const HEADER_WORDS: usize = 32;
/// Bytes of both headers.
const HEADERS: usize = HEADER_WORDS * 4;
const SEG_BYTES: usize = 32;
const FLAG_DIRECT: u32 = 1;
/// The rows after the first take the tokens the model drafts itself.
const FLAG_OWN_DRAFTS: u32 = 2;
/// The rows rebuild recurrent state from cached cells and append nothing.
const FLAG_REPLAY: u32 = 4;

// Word indices in the runtime header.
const H_MAGIC: usize = 8;
const H_STEP: usize = 9;
const H_RUNS: usize = 10;
const H_UPDATES: usize = 11;
const H_ENTRIES: usize = 12;
const H_COMMITS: usize = 13;
const H_OFF_EXTRA: usize = 14;
const H_OFF_RUNS: usize = 15;
const H_OFF_COMMIT: usize = 16;
const H_OFF_UPDATES: usize = 17;
const H_OFF_ENTRIES: usize = 18;
const H_MAX_SEG_ROWS: usize = 19;
const H_MAX_COMMIT_ROWS: usize = 20;
const H_OFF_SAMPLERS: usize = 21;
const H_STATE_COPIES: usize = 22;
const H_OFF_STATE_COPIES: usize = 23;
const H_PAGE_COPIES: usize = 24;
const H_OFF_PAGE_COPIES: usize = 25;
const H_CORRECTIONS: usize = 26;
const H_OFF_CORRECTIONS: usize = 27;
const H_MASKS: usize = 28;
const H_OFF_MASKS: usize = 29;
const H_MASK_WORDS: usize = 30;

/// One row of a step: a token of a sequence at a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    /// The sequence's slot, which indexes its page table on the devices.
    pub slot: u32,
    pub position: u32,
    pub token: u32,
    /// See the module documentation.
    pub context: u32,
}

/// A change to a sequence's page table on the devices: entries
/// `first..first + count` of its table are the next `count` entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableUpdate {
    pub slot: u32,
    pub first: u32,
    pub count: u32,
}

/// What a state copy does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateCopyKind {
    /// The slot's recurrent state goes into the place.
    Save,
    /// The place goes into the slot's recurrent state.
    Restore,
}

/// A copy of the recurrent state of the sequence in `slot` to or from a
/// checkpoint place; see the module documentation for where it runs in a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCopy {
    pub kind: StateCopyKind,
    pub slot: u32,
    pub place: u32,
}

/// The first `cells` cells of page `from` go into page `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageCopy {
    pub from: u32,
    pub to: u32,
    pub cells: u32,
}

/// The outcome of a draft window of the step before: of the rows
/// `row0..row0 + rows` that step ran for the sequence in `slot`, the first
/// `accepted` are committed to its recurrent state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commit {
    pub row0: u32,
    pub rows: u32,
    pub slot: u32,
    pub accepted: u32,
}

fn align(n: usize) -> usize {
    n.next_multiple_of(16)
}

/// The bytes the record can take at most, for buffer sizes: a step of up to
/// `max_rows` rows of at most `max_slots` sequences, of which `wanted` rows
/// at most have their logits wanted, with a vocabulary of `vocab` tokens.
pub fn max_bytes(max_rows: u32, max_slots: u32, wanted: u32, vocab: u32) -> usize {
    let (rows, slots) = (max_rows as usize, max_slots as usize);
    let wanted = (wanted as usize).min(rows);
    // A segment and a commit per sequence at worst, a run per wanted row; a
    // step creates at most one page per row and re-dirties at most one entry
    // per sequence, and a sequence has at most one update. A slot has at most
    // a save at a position it reached, a save of the state it ends with and
    // the restore of the sequence that takes it next, between two records; a
    // sequence copies at most one page. Every wanted row may have the most
    // corrections and a mask.
    let segments = slots.min(rows.max(1));
    let bytes = HEADERS
        + SEG_BYTES * segments
        + rows * (4 + 16 + 8)
        + wanted * (4 + 8 + SampleRow::BYTES + ENTRIES_PER_ROW * Entry::BYTES)
        + wanted * TokenMask::words_for(vocab) * 4
        + KERNEL_HEADER
        + SEG_BYTES * slots
        + slots * 12
        + (rows + 2 * slots) * 4
        + slots * 3 * 12
        + segments * 12
        + 16 * 8;
    bytes.next_multiple_of(256)
}

/// The record does not fit its buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the step record does not fit its {capacity}-byte buffer")]
pub struct RecordFull {
    pub capacity: usize,
}

/// Writes a record into a buffer: the segments with their rows, then the
/// runs, the commits, the updates and the entries, in that order.
#[derive(Debug)]
pub struct RecordWriter<'a> {
    buf: &'a mut [u8],
    n_segs: u32,
    n_rows: u32,
    segs_written: u32,
    rows_written: u32,
    off_tokens: usize,
    off_pos: usize,
    off_extra: usize,
    /// The end of what is written after the row arrays.
    end: usize,
    header: [u32; HEADER_WORDS],
}

impl<'a> RecordWriter<'a> {
    /// Starts a record of `n_segs` segments with `n_rows` rows in all.
    pub fn new(buf: &'a mut [u8], step: u64, n_segs: u32, n_rows: u32) -> Result<Self, RecordFull> {
        let off_segs = HEADERS;
        let off_tokens = align(off_segs + n_segs as usize * SEG_BYTES);
        let off_pos = align(off_tokens + n_rows as usize * 4);
        let off_extra = align(off_pos + n_rows as usize * 16);
        let end = align(off_extra + n_rows as usize * 8);
        if end > buf.len() {
            return Err(RecordFull { capacity: buf.len() });
        }
        let mut header = [0u32; HEADER_WORDS];
        header[0] = n_rows;
        header[1] = n_segs;
        header[4] = off_segs as u32;
        header[5] = off_tokens as u32;
        header[6] = off_pos as u32;
        header[7] = end as u32;
        header[H_MAGIC] = MAGIC;
        header[H_STEP] = step as u32;
        header[H_OFF_EXTRA] = off_extra as u32;
        for h in [
            H_OFF_RUNS,
            H_OFF_SAMPLERS,
            H_OFF_COMMIT,
            H_OFF_UPDATES,
            H_OFF_ENTRIES,
            H_OFF_STATE_COPIES,
            H_OFF_PAGE_COPIES,
            H_OFF_CORRECTIONS,
            H_OFF_MASKS,
        ] {
            header[h] = end as u32;
        }
        Ok(Self {
            buf,
            n_segs,
            n_rows,
            segs_written: 0,
            rows_written: 0,
            off_tokens,
            off_pos,
            off_extra,
            end,
            header,
        })
    }

    fn put(&mut self, at: usize, words: &[u32]) -> Result<(), RecordFull> {
        let capacity = self.buf.len();
        let dst = self.buf.get_mut(at..at + words.len() * 4).ok_or(RecordFull { capacity })?;
        for (chunk, word) in dst.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        Ok(())
    }

    /// The next segment: `rows` consecutive positions of the sequence in
    /// `slot` from `position`. Its rows follow with [`RecordWriter::row`].
    pub fn segment(
        &mut self,
        slot: u32,
        position: u32,
        rows: u32,
        direct: bool,
    ) -> Result<(), RecordFull> {
        assert!(self.segs_written < self.n_segs, "more segments than the record was started for");
        let at = HEADERS + self.segs_written as usize * SEG_BYTES;
        let row0 = self.rows_written;
        let flags = if direct { FLAG_DIRECT } else { 0 };
        self.put(at, &[row0, rows, slot, rows, position, flags, 0, 0])?;
        self.segs_written += 1;
        self.header[H_MAX_SEG_ROWS] = self.header[H_MAX_SEG_ROWS].max(rows);
        Ok(())
    }

    /// Marks the segment written last as a window whose draft rows take the
    /// tokens the model drafts itself: their tokens in this record are
    /// placeholders.
    pub fn own_drafts(&mut self) -> Result<(), RecordFull> {
        assert!(self.segs_written > 0, "a segment was written");
        let at = HEADERS + (self.segs_written as usize - 1) * SEG_BYTES + 20;
        let capacity = self.buf.len();
        let word = self.buf.get_mut(at..at + 4).ok_or(RecordFull { capacity })?;
        let flags = u32::from_le_bytes([word[0], word[1], word[2], word[3]]) | FLAG_OWN_DRAFTS;
        word.copy_from_slice(&flags.to_le_bytes());
        Ok(())
    }

    /// Marks the segment written last as a replay: its rows rebuild recurrent
    /// state from cells that are cached already and append nothing. A replay
    /// is direct.
    pub fn replay(&mut self) -> Result<(), RecordFull> {
        assert!(self.segs_written > 0, "a segment was written");
        let at = HEADERS + (self.segs_written as usize - 1) * SEG_BYTES + 20;
        let capacity = self.buf.len();
        let word = self.buf.get_mut(at..at + 4).ok_or(RecordFull { capacity })?;
        let flags =
            u32::from_le_bytes([word[0], word[1], word[2], word[3]]) | FLAG_REPLAY | FLAG_DIRECT;
        word.copy_from_slice(&flags.to_le_bytes());
        Ok(())
    }

    /// The next row of the segment being written.
    pub fn row(&mut self, row: Row) -> Result<(), RecordFull> {
        assert!(self.rows_written < self.n_rows, "more rows than the record was started for");
        let i = self.rows_written as usize;
        let p = row.position;
        self.put(self.off_tokens + i * 4, &[row.token])?;
        self.put(self.off_pos + i * 16, &[p, p, p, 0])?;
        self.put(self.off_extra + i * 8, &[row.slot, row.context])?;
        self.rows_written += 1;
        Ok(())
    }

    /// The rows whose logits are wanted, as ranges `(first row, rows)` in
    /// order.
    pub fn runs(&mut self, runs: &[(u32, u32)]) -> Result<(), RecordFull> {
        let off_out = self.end;
        let mut at = off_out;
        let mut picked = 0;
        for &(first, count) in runs {
            for row in first..first + count {
                self.put(at, &[row])?;
                at += 4;
            }
            picked += count;
        }
        let off_runs = align(at);
        at = off_runs;
        for &(first, count) in runs {
            self.put(at, &[first, count])?;
            at += 8;
        }
        self.header[2] = picked;
        self.header[7] = off_out as u32;
        self.header[H_RUNS] = runs.len() as u32;
        self.header[H_OFF_RUNS] = off_runs as u32;
        // Every wanted row takes the most probable token until
        // `RecordWriter::samplers` says otherwise.
        let off_samplers = align(at);
        let greedy = SampleParams::GREEDY.row(0).to_bytes();
        let capacity = self.buf.len();
        at = off_samplers;
        for _ in 0..picked {
            self.buf
                .get_mut(at..at + SampleRow::BYTES)
                .ok_or(RecordFull { capacity })?
                .copy_from_slice(&greedy);
            at += SampleRow::BYTES;
        }
        self.header[H_OFF_SAMPLERS] = off_samplers as u32;
        self.end = align(at);
        Ok(())
    }

    /// How each wanted row is picked, in the order of the wanted rows; after
    /// [`RecordWriter::runs`]. Rows this is not called for take the most
    /// probable token.
    pub fn samplers(
        &mut self,
        rows: impl IntoIterator<Item = SampleRow>,
    ) -> Result<(), RecordFull> {
        let picked = self.header[2] as usize;
        let mut at = self.header[H_OFF_SAMPLERS] as usize;
        let capacity = self.buf.len();
        for (i, row) in rows.into_iter().enumerate() {
            assert!(i < picked, "more sampler rows than wanted rows");
            self.buf
                .get_mut(at..at + SampleRow::BYTES)
                .ok_or(RecordFull { capacity })?
                .copy_from_slice(&row.to_bytes());
            at += SampleRow::BYTES;
        }
        Ok(())
    }

    /// The logit corrections of the wanted rows, one list after the other: a
    /// sampler row names its own as a range of these.
    pub fn corrections(&mut self, entries: &[Entry]) -> Result<(), RecordFull> {
        let off = self.end;
        let mut at = off;
        let capacity = self.buf.len();
        for entry in entries {
            self.buf
                .get_mut(at..at + Entry::BYTES)
                .ok_or(RecordFull { capacity })?
                .copy_from_slice(&entry.to_bytes());
            at += Entry::BYTES;
        }
        self.header[H_CORRECTIONS] = entries.len() as u32;
        self.header[H_OFF_CORRECTIONS] = off as u32;
        self.end = align(at);
        Ok(())
    }

    /// The masks of the wanted rows: `words` words each, a bit per token; a
    /// sampler row names its own by its number here.
    pub fn masks(&mut self, masks: &[u32], words: usize) -> Result<(), RecordFull> {
        let off = self.end;
        let capacity = self.buf.len();
        let dst = self.buf.get_mut(off..off + masks.len() * 4).ok_or(RecordFull { capacity })?;
        for (chunk, word) in dst.as_chunks_mut::<4>().0.iter_mut().zip(masks) {
            *chunk = word.to_le_bytes();
        }
        self.header[H_MASKS] = masks.len().checked_div(words).unwrap_or(0) as u32;
        self.header[H_MASK_WORDS] = words as u32;
        self.header[H_OFF_MASKS] = off as u32;
        self.end = align(off + masks.len() * 4);
        Ok(())
    }

    /// The outcomes of the draft windows of the step before.
    pub fn commits(&mut self, commits: &[Commit]) -> Result<(), RecordFull> {
        if commits.is_empty() {
            return Ok(());
        }
        let off = self.end;
        let mut longest = 0;
        self.put(off, &[0, commits.len() as u32, 0, 0, KERNEL_HEADER as u32, 0, 0, 0])?;
        let mut at = off + KERNEL_HEADER;
        for c in commits {
            self.put(at, &[c.row0, c.rows, c.slot, c.accepted.min(c.rows), 0, 0, 0, 0])?;
            longest = longest.max(c.rows);
            at += SEG_BYTES;
        }
        self.header[H_COMMITS] = commits.len() as u32;
        self.header[H_OFF_COMMIT] = off as u32;
        self.header[H_MAX_COMMIT_ROWS] = longest;
        self.end = align(at);
        Ok(())
    }

    /// The changes to the page tables: every update, then the entries of all
    /// updates in order.
    pub fn updates(
        &mut self,
        updates: impl Iterator<Item = TableUpdate>,
        entries: impl Iterator<Item = u32>,
    ) -> Result<(), RecordFull> {
        let off_updates = self.end;
        let mut at = off_updates;
        let mut n_updates = 0;
        for u in updates {
            self.put(at, &[u.slot, u.first, u.count])?;
            at += 12;
            n_updates += 1;
        }
        let off_entries = align(at);
        at = off_entries;
        let mut n_entries = 0;
        for e in entries {
            self.put(at, &[e])?;
            at += 4;
            n_entries += 1;
        }
        self.header[H_UPDATES] = n_updates;
        self.header[H_ENTRIES] = n_entries;
        self.header[H_OFF_UPDATES] = off_updates as u32;
        self.header[H_OFF_ENTRIES] = off_entries as u32;
        self.end = align(at);
        Ok(())
    }

    /// The copies of recurrent state and of pages that run before the rows of
    /// this step, in the order the module documentation gives.
    pub fn copies(&mut self, states: &[StateCopy], pages: &[PageCopy]) -> Result<(), RecordFull> {
        let off_states = self.end;
        let mut at = off_states;
        for copy in states {
            let kind = match copy.kind {
                StateCopyKind::Save => 0,
                StateCopyKind::Restore => 1,
            };
            self.put(at, &[kind, copy.slot, copy.place])?;
            at += 12;
        }
        let off_pages = align(at);
        at = off_pages;
        for copy in pages {
            self.put(at, &[copy.from, copy.to, copy.cells])?;
            at += 12;
        }
        self.header[H_STATE_COPIES] = states.len() as u32;
        self.header[H_OFF_STATE_COPIES] = off_states as u32;
        self.header[H_PAGE_COPIES] = pages.len() as u32;
        self.header[H_OFF_PAGE_COPIES] = off_pages as u32;
        self.end = align(at);
        Ok(())
    }

    /// Writes the headers and returns the bytes used, a multiple of 16 so the
    /// copy is a whole number of the device's transfer units.
    pub fn finish(mut self) -> Result<usize, RecordFull> {
        assert_eq!(self.segs_written, self.n_segs, "every segment of the record is written");
        assert_eq!(self.rows_written, self.n_rows, "every row of the record is written");
        let header = self.header;
        self.put(0, &header)?;
        // Padding words keep whatever the last step left there; nothing reads
        // them, because every section is counted in the headers.
        Ok(self.end)
    }
}

/// Why bytes are not a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RecordError {
    #[error("the buffer is too short for a record header")]
    Short,
    #[error("the buffer does not start with a step record")]
    BadMagic,
    #[error("the header counts more than the buffer holds")]
    Truncated,
}

/// Reads a record.
#[derive(Clone, Copy, Debug)]
pub struct RecordView<'a> {
    bytes: &'a [u8],
    header: [u32; HEADER_WORDS],
}

impl<'a> RecordView<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, RecordError> {
        if bytes.len() < HEADERS {
            return Err(RecordError::Short);
        }
        let mut header = [0u32; HEADER_WORDS];
        for (slot, chunk) in header.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *slot = u32::from_le_bytes(*chunk);
        }
        if header[H_MAGIC] != MAGIC {
            return Err(RecordError::BadMagic);
        }
        let view = Self { bytes, header };
        let (rows, segs) = (header[0] as usize, header[1] as usize);
        let ends = [
            header[4] as usize + segs * SEG_BYTES,
            header[5] as usize + rows * 4,
            header[6] as usize + rows * 16,
            header[7] as usize + header[2] as usize * 4,
            header[H_OFF_EXTRA] as usize + rows * 8,
            header[H_OFF_RUNS] as usize + header[H_RUNS] as usize * 8,
            header[H_OFF_SAMPLERS] as usize + header[2] as usize * SampleRow::BYTES,
            header[H_OFF_COMMIT] as usize
                + if header[H_COMMITS] > 0 {
                    KERNEL_HEADER + header[H_COMMITS] as usize * SEG_BYTES
                } else {
                    0
                },
            header[H_OFF_UPDATES] as usize + header[H_UPDATES] as usize * 12,
            header[H_OFF_ENTRIES] as usize + header[H_ENTRIES] as usize * 4,
            header[H_OFF_STATE_COPIES] as usize + header[H_STATE_COPIES] as usize * 12,
            header[H_OFF_PAGE_COPIES] as usize + header[H_PAGE_COPIES] as usize * 12,
            header[H_OFF_CORRECTIONS] as usize + header[H_CORRECTIONS] as usize * Entry::BYTES,
            header[H_OFF_MASKS] as usize
                + header[H_MASKS] as usize * header[H_MASK_WORDS] as usize * 4,
        ];
        if ends.iter().any(|end| *end > bytes.len()) {
            return Err(RecordError::Truncated);
        }
        Ok(view)
    }

    fn word(&self, byte: usize) -> u32 {
        u32::from_le_bytes(self.bytes[byte..byte + 4].try_into().expect("4 bytes"))
    }

    /// Overwrites the token and the context of row `row` in the bytes of a
    /// record, as the device's draft chain overwrites the tokens of the draft
    /// rows of a window before the model's blocks read them. For an executor
    /// that stands in for a device.
    pub fn set_row_inputs(bytes: &mut [u8], row: u32, token: u32, context: u32) {
        let word = |bytes: &[u8], at: usize| {
            u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes")) as usize
        };
        let (tokens, extra) = (word(bytes, 5 * 4), word(bytes, H_OFF_EXTRA * 4));
        let (at_token, at_context) = (tokens + row as usize * 4, extra + row as usize * 8 + 4);
        bytes[at_token..at_token + 4].copy_from_slice(&token.to_le_bytes());
        bytes[at_context..at_context + 4].copy_from_slice(&context.to_le_bytes());
    }

    /// The low 32 bits of the step number.
    pub fn step(&self) -> u32 {
        self.header[H_STEP]
    }

    pub fn rows(&self) -> u32 {
        self.header[0]
    }

    pub fn segments(&self) -> u32 {
        self.header[1]
    }

    pub fn runs(&self) -> u32 {
        self.header[H_RUNS]
    }

    pub fn updates(&self) -> u32 {
        self.header[H_UPDATES]
    }

    pub fn entries(&self) -> u32 {
        self.header[H_ENTRIES]
    }

    /// The number of rows whose logits are wanted (the sum of the runs).
    pub fn picked_rows(&self) -> u32 {
        self.header[2]
    }

    /// The sizes the layer kernels' grids follow from.
    pub fn dims(&self) -> StepDims {
        StepDims {
            n_rows: self.header[0],
            n_segs: self.header[1],
            max_seg_rows: self.header[H_MAX_SEG_ROWS],
            n_out: self.header[2],
        }
    }

    fn segment_at(&self, at: usize) -> (Segment, u32) {
        let segment = Segment {
            row0: self.word(at),
            n_rows: self.word(at + 4),
            slot: self.word(at + 8),
            pos0: self.word(at + 16) as i32,
            direct: self.word(at + 20) & FLAG_DIRECT != 0,
        };
        (segment, self.word(at + 12))
    }

    /// Segment `i` of the step.
    pub fn segment(&self, i: u32) -> Segment {
        assert!(i < self.segments(), "segment {i} of {}", self.segments());
        self.segment_at(self.header[4] as usize + i as usize * SEG_BYTES).0
    }

    /// Whether segment `i` is a window whose draft rows take the tokens the
    /// model drafts itself.
    pub fn own_drafts(&self, i: u32) -> bool {
        assert!(i < self.segments(), "segment {i} of {}", self.segments());
        self.word(self.header[4] as usize + i as usize * SEG_BYTES + 20) & FLAG_OWN_DRAFTS != 0
    }

    /// Whether segment `i` is a replay: its rows rebuild recurrent state from
    /// cached cells and append nothing.
    pub fn replay(&self, i: u32) -> bool {
        assert!(i < self.segments(), "segment {i} of {}", self.segments());
        self.word(self.header[4] as usize + i as usize * SEG_BYTES + 20) & FLAG_REPLAY != 0
    }

    /// The byte offset of the token array (a u32 per row) in the record.
    pub fn tokens_offset(&self) -> u64 {
        u64::from(self.header[5])
    }

    /// The byte offset of the sampler rows ([`SampleRow::BYTES`] bytes per
    /// wanted row) in the record.
    pub fn samplers_offset(&self) -> u64 {
        u64::from(self.header[H_OFF_SAMPLERS])
    }

    /// How wanted row `i` (in the order of the runs) is picked.
    pub fn sampler(&self, i: u32) -> SampleRow {
        assert!(i < self.picked_rows(), "wanted row {i} of {}", self.picked_rows());
        let at = self.header[H_OFF_SAMPLERS] as usize + i as usize * SampleRow::BYTES;
        SampleRow::from_bytes(self.bytes[at..at + SampleRow::BYTES].try_into().expect("a row"))
    }

    /// The byte offset of the corrections ([`Entry::BYTES`] bytes each) in
    /// the record.
    pub fn corrections_offset(&self) -> u64 {
        u64::from(self.header[H_OFF_CORRECTIONS])
    }

    /// The number of corrections of all rows.
    pub fn corrections(&self) -> u32 {
        self.header[H_CORRECTIONS]
    }

    /// Correction `i` of the record.
    pub fn correction(&self, i: u32) -> Entry {
        assert!(i < self.corrections(), "correction {i} of {}", self.corrections());
        let at = self.header[H_OFF_CORRECTIONS] as usize + i as usize * Entry::BYTES;
        let f = |k: usize| f32::from_bits(self.word(at + 4 * k));
        Entry { token: self.word(at), bias: f(1), rep: f(2), sub: f(3) }
    }

    /// The byte offset of the masks in the record.
    pub fn masks_offset(&self) -> u64 {
        u64::from(self.header[H_OFF_MASKS])
    }

    /// The number of masks, and the words of each.
    pub fn masks(&self) -> (u32, u32) {
        (self.header[H_MASKS], self.header[H_MASK_WORDS])
    }

    /// Word `word` of mask `slot`.
    pub fn mask_word(&self, slot: u32, word: u32) -> u32 {
        let (count, words) = self.masks();
        assert!(slot < count && word < words, "word {word} of mask {slot} ({count} of {words})");
        self.word(
            self.header[H_OFF_MASKS] as usize
                + (slot as usize * words as usize + word as usize) * 4,
        )
    }

    pub fn row(&self, i: u32) -> Row {
        assert!(i < self.rows(), "row {i} of {}", self.rows());
        let i = i as usize;
        let extra = self.header[H_OFF_EXTRA] as usize + i * 8;
        Row {
            slot: self.word(extra),
            position: self.word(self.header[6] as usize + i * 16),
            token: self.word(self.header[5] as usize + i * 4),
            context: self.word(extra + 4),
        }
    }

    /// `(first row, rows)` of run `i`.
    pub fn run(&self, i: u32) -> (u32, u32) {
        assert!(i < self.runs(), "run {i} of {}", self.runs());
        let at = self.header[H_OFF_RUNS] as usize + i as usize * 8;
        (self.word(at), self.word(at + 4))
    }

    /// The number of draft windows of the step before whose outcome this
    /// record carries.
    pub fn commits(&self) -> u32 {
        self.header[H_COMMITS]
    }

    /// Outcome `i` of the step before.
    pub fn commit(&self, i: u32) -> Commit {
        assert!(i < self.commits(), "commit {i} of {}", self.commits());
        let at = self.header[H_OFF_COMMIT] as usize + KERNEL_HEADER + i as usize * SEG_BYTES;
        let (segment, accepted) = self.segment_at(at);
        Commit { row0: segment.row0, rows: segment.n_rows, slot: segment.slot, accepted }
    }

    /// The commit record, when the step before had draft windows: its byte
    /// offset in the record (its device address is the `step` argument of the
    /// commit kernels) and the sizes of its launch.
    pub fn commit_record(&self) -> Option<(u64, StepDims)> {
        (self.commits() > 0).then(|| {
            (
                u64::from(self.header[H_OFF_COMMIT]),
                StepDims {
                    n_rows: 0,
                    n_segs: self.commits(),
                    max_seg_rows: self.header[H_MAX_COMMIT_ROWS],
                    n_out: 0,
                },
            )
        })
    }

    /// The number of copies of recurrent state this step starts with.
    pub fn state_copies(&self) -> u32 {
        self.header[H_STATE_COPIES]
    }

    /// State copy `i`, in the order they run.
    pub fn state_copy(&self, i: u32) -> StateCopy {
        assert!(i < self.state_copies(), "state copy {i} of {}", self.state_copies());
        let at = self.header[H_OFF_STATE_COPIES] as usize + i as usize * 12;
        let kind = match self.word(at) {
            0 => StateCopyKind::Save,
            _ => StateCopyKind::Restore,
        };
        StateCopy { kind, slot: self.word(at + 4), place: self.word(at + 8) }
    }

    /// The number of page copies this step starts with.
    pub fn page_copies(&self) -> u32 {
        self.header[H_PAGE_COPIES]
    }

    /// Page copy `i`, in the order they run.
    pub fn page_copy(&self, i: u32) -> PageCopy {
        assert!(i < self.page_copies(), "page copy {i} of {}", self.page_copies());
        let at = self.header[H_OFF_PAGE_COPIES] as usize + i as usize * 12;
        PageCopy { from: self.word(at), to: self.word(at + 4), cells: self.word(at + 8) }
    }

    pub fn update(&self, i: u32) -> TableUpdate {
        assert!(i < self.updates(), "update {i} of {}", self.updates());
        let at = self.header[H_OFF_UPDATES] as usize + i as usize * 12;
        TableUpdate { slot: self.word(at), first: self.word(at + 4), count: self.word(at + 8) }
    }

    /// The word offset of the entries section, in 32-bit words from the start
    /// of the record: the device copy of an update's entries is there.
    pub fn entries_offset_words(&self) -> usize {
        self.header[H_OFF_ENTRIES] as usize / 4
    }

    pub fn entry(&self, i: u32) -> u32 {
        assert!(i < self.entries(), "entry {i} of {}", self.entries());
        self.word(self.header[H_OFF_ENTRIES] as usize + i as usize * 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Vec<u8>, usize) {
        let mut buf = vec![0xAAu8; 1024];
        let mut w = RecordWriter::new(&mut buf, 77, 2, 3).unwrap();
        w.segment(3, 10, 2, false).unwrap();
        w.row(Row { slot: 3, position: 10, token: 65, context: 900 }).unwrap();
        w.row(Row { slot: 3, position: 11, token: 66, context: 901 }).unwrap();
        w.segment(5, 0, 1, true).unwrap();
        w.row(Row { slot: 5, position: 0, token: 67, context: 902 }).unwrap();
        w.runs(&[(0, 2), (2, 1)]).unwrap();
        w.commits(&[Commit { row0: 4, rows: 3, slot: 3, accepted: 2 }]).unwrap();
        w.updates([TableUpdate { slot: 3, first: 1, count: 2 }].into_iter(), [40, 41].into_iter())
            .unwrap();
        let used = w.finish().unwrap();
        (buf, used)
    }

    #[test]
    fn a_record_reads_back_exactly_what_was_written() {
        let (buf, used) = sample();
        assert_eq!(used % 16, 0);
        let record = RecordView::new(&buf[..used]).unwrap();
        assert_eq!((record.step(), record.rows(), record.runs()), (77, 3, 2));
        assert_eq!((record.updates(), record.entries(), record.picked_rows()), (1, 2, 3));
        assert_eq!(record.row(1), Row { slot: 3, position: 11, token: 66, context: 901 });
        assert_eq!(record.row(2).slot, 5);
        assert_eq!((record.run(0), record.run(1)), ((0, 2), (2, 1)));
        assert_eq!(record.update(0), TableUpdate { slot: 3, first: 1, count: 2 });
        assert_eq!((record.entry(0), record.entry(1)), (40, 41));
        assert_eq!(
            record.segment(0),
            Segment { row0: 0, n_rows: 2, slot: 3, pos0: 10, direct: false }
        );
        assert_eq!(
            record.segment(1),
            Segment { row0: 2, n_rows: 1, slot: 5, pos0: 0, direct: true }
        );
        assert_eq!(record.dims(), StepDims { n_rows: 3, n_segs: 2, max_seg_rows: 2, n_out: 3 });
        assert_eq!(record.commit(0), Commit { row0: 4, rows: 3, slot: 3, accepted: 2 });
        let (_, dims) = record.commit_record().unwrap();
        assert_eq!((dims.n_segs, dims.max_seg_rows), (1, 3));
    }

    #[test]
    fn the_wanted_rows_carry_how_they_are_picked() {
        // Without a word about it every wanted row takes the most probable token.
        let (buf, used) = sample();
        let record = RecordView::new(&buf[..used]).unwrap();
        let greedy = SampleParams::GREEDY.row(0);
        assert!((0..3).all(|i| record.sampler(i) == greedy && !record.sampler(i).sampled()));
        assert!(!record.own_drafts(0) && !record.own_drafts(1));

        // A window whose drafts the model makes itself, sampled.
        let params = SampleParams {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            seed: 7,
            ..SampleParams::GREEDY
        };
        let mut buf = vec![0x55u8; 1024];
        let mut w = RecordWriter::new(&mut buf, 1, 2, 4).unwrap();
        w.segment(0, 40, 3, false).unwrap();
        w.own_drafts().unwrap();
        w.row(Row { slot: 0, position: 40, token: 9, context: 0 }).unwrap();
        w.row(Row { slot: 0, position: 41, token: 0, context: 0 }).unwrap();
        w.row(Row { slot: 0, position: 42, token: 0, context: 0 }).unwrap();
        w.segment(1, 7, 1, true).unwrap();
        w.row(Row { slot: 1, position: 7, token: 5, context: 0 }).unwrap();
        w.runs(&[(0, 4)]).unwrap();
        w.samplers((40..43).map(|p| params.row(p))).unwrap();
        let used = w.finish().unwrap();
        let record = RecordView::new(&buf[..used]).unwrap();
        assert!(record.own_drafts(0) && !record.own_drafts(1));
        // The flag is not the kernels' "direct" flag.
        assert!(!record.segment(0).direct && record.segment(1).direct);
        assert_eq!(record.sampler(1), params.row(41));
        assert!(record.sampler(2).sampled());
        assert_eq!(record.sampler(3), greedy, "a row nobody set stays the most probable token");
        // The token array is where the device writes the drafts.
        let at = record.tokens_offset() as usize;
        assert_eq!(u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()), 9);
        let rows = record.samplers_offset() as usize;
        assert_eq!(
            buf[rows + SampleRow::BYTES..rows + 2 * SampleRow::BYTES],
            params.row(41).to_bytes()
        );
    }

    /// The first bytes are the record the layer kernels read: the same bytes
    /// `ares_cuda::StepRecord` builds for the same step, section by section.
    #[test]
    fn the_start_of_a_record_is_the_layer_kernels_record() {
        let (buf, used) = sample();
        let segments = [
            Segment { row0: 0, n_rows: 2, slot: 3, pos0: 10, direct: false },
            Segment { row0: 2, n_rows: 1, slot: 5, pos0: 0, direct: true },
        ];
        let theirs =
            ares_cuda::StepRecord::new(&segments, &[65, 66, 67], None, &[0, 1, 2]).unwrap();
        let theirs = theirs.as_bytes();
        let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        let ours = &buf[..used];
        // Counts.
        for at in [0, 4, 8] {
            assert_eq!(word(ours, at), word(theirs, at));
        }
        // Each array, found through its offset in either record.
        for (field, bytes) in [(16, 2 * 32), (20, 3 * 4), (24, 3 * 16), (28, 3 * 4)] {
            let (a, b) = (word(ours, field) as usize, word(theirs, field) as usize);
            assert_eq!(a % 16, 0, "arrays are aligned");
            assert_eq!(ours[a..a + bytes], theirs[b..b + bytes], "array at header byte {field}");
        }
        // The commit record is one too: a header and its segments.
        let record = RecordView::new(ours).unwrap();
        let (at, _) = record.commit_record().unwrap();
        let commit = &ours[at as usize..];
        assert_eq!((word(commit, 4), word(commit, 16)), (1, 32));
        assert_eq!(
            [word(commit, 32), word(commit, 36), word(commit, 40), word(commit, 44)],
            [4, 3, 3, 2]
        );
        assert_eq!(word(commit, 52), 0, "a commit segment is not direct");
    }

    #[test]
    fn a_record_that_does_not_fit_is_an_error_and_not_a_panic() {
        let mut buf = vec![0u8; HEADERS + 32 + 16 + 16];
        assert!(RecordWriter::new(&mut buf, 1, 1, 1).is_err());
        let mut buf = vec![0u8; HEADERS + 32 + 16 + 16 + 16];
        let mut w = RecordWriter::new(&mut buf, 1, 1, 1).unwrap();
        w.segment(0, 0, 1, true).unwrap();
        w.row(Row { slot: 0, position: 0, token: 0, context: 0 }).unwrap();
        let full = w.runs(&[(0, 1)]).unwrap_err();
        assert_eq!(full.capacity, buf.len());
    }

    #[test]
    fn bytes_that_are_not_a_record_are_refused() {
        assert_eq!(RecordView::new(&[0; 8]).unwrap_err(), RecordError::Short);
        assert_eq!(RecordView::new(&[0; 128]).unwrap_err(), RecordError::BadMagic);
        let (buf, used) = sample();
        assert_eq!(RecordView::new(&buf[..used - 16]).unwrap_err(), RecordError::Truncated);
    }

    #[test]
    fn the_buffer_size_covers_a_full_step() {
        // 64 rows of 64 sequences, every row wanted on its own, each sequence
        // with a new page, a commit and a table update: the worst case the
        // planner can build.
        let bytes = max_bytes(64, 256, 64, 248_320);
        let mut buf = vec![0u8; bytes];
        let mut w = RecordWriter::new(&mut buf, 1, 64, 64).unwrap();
        for i in 0..64 {
            w.segment(i, 0, 1, false).unwrap();
            w.row(Row { slot: i, position: 0, token: 1, context: 0 }).unwrap();
        }
        let runs: Vec<(u32, u32)> = (0..64).map(|r| (r, 1)).collect();
        w.runs(&runs).unwrap();
        let commits: Vec<Commit> =
            (0..256).map(|s| Commit { row0: 0, rows: 1, slot: s, accepted: 1 }).collect();
        w.commits(&commits).unwrap();
        w.updates((0..256).map(|slot| TableUpdate { slot, first: 0, count: 1 }), 0..(64 + 2 * 256))
            .unwrap();
        // Every slot saves twice and restores once, and every sequence of the
        // step copies a page.
        let states: Vec<StateCopy> = (0..256)
            .flat_map(|slot| {
                [StateCopyKind::Save, StateCopyKind::Save, StateCopyKind::Restore]
                    .map(|kind| StateCopy { kind, slot, place: slot })
            })
            .collect();
        let pages: Vec<PageCopy> =
            (0..64).map(|i| PageCopy { from: i, to: 64 + i, cells: 3 }).collect();
        w.copies(&states, &pages).unwrap();
        assert!(w.finish().unwrap() <= bytes);
    }

    #[test]
    fn replay_segments_and_the_copies_of_a_step_read_back_as_written() {
        let mut buf = vec![0x77u8; 2048];
        let mut w = RecordWriter::new(&mut buf, 5, 3, 4).unwrap();
        w.segment(2, 32, 2, true).unwrap();
        w.replay().unwrap();
        w.row(Row { slot: 2, position: 32, token: 1, context: 1 }).unwrap();
        w.row(Row { slot: 2, position: 33, token: 2, context: 2 }).unwrap();
        w.segment(3, 40, 1, false).unwrap();
        w.row(Row { slot: 3, position: 40, token: 3, context: 3 }).unwrap();
        w.segment(4, 0, 1, true).unwrap();
        w.row(Row { slot: 4, position: 0, token: 4, context: 4 }).unwrap();
        w.runs(&[(2, 2)]).unwrap();
        let states = [
            StateCopy { kind: StateCopyKind::Save, slot: 3, place: 9 },
            StateCopy { kind: StateCopyKind::Restore, slot: 2, place: 4 },
        ];
        let pages = [PageCopy { from: 12, to: 30, cells: 5 }];
        w.copies(&states, &pages).unwrap();
        let with_copies = w.finish().unwrap();
        let record = RecordView::new(&buf[..with_copies]).unwrap();
        assert_eq!(with_copies % 16, 0);
        assert!(record.replay(0) && !record.replay(1) && !record.replay(2));
        assert!(record.segment(0).direct, "a replay is direct");
        assert!(!record.own_drafts(0));
        assert_eq!((record.state_copies(), record.page_copies()), (2, 1));
        assert_eq!((record.state_copy(0), record.state_copy(1)), (states[0], states[1]));
        assert_eq!(record.page_copy(0), pages[0]);

        // A record without copies has none, and one cut short of them is refused.
        let (plain, used) = sample();
        let plain = RecordView::new(&plain[..used]).unwrap();
        assert_eq!((plain.state_copies(), plain.page_copies()), (0, 0));
        assert!(!plain.replay(0));
        assert_eq!(RecordView::new(&buf[..with_copies - 16]).unwrap_err(), RecordError::Truncated);
    }

    #[test]
    fn corrections_and_masks_read_back_as_written() {
        let mut buf = vec![0x66u8; 4096];
        let mut w = RecordWriter::new(&mut buf, 3, 1, 2).unwrap();
        w.segment(0, 10, 2, false).unwrap();
        w.row(Row { slot: 0, position: 10, token: 4, context: 0 }).unwrap();
        w.row(Row { slot: 0, position: 11, token: 5, context: 0 }).unwrap();
        w.runs(&[(0, 2)]).unwrap();
        let entries = [
            Entry { token: 7, bias: -1.5, rep: 1.25, sub: 0.5 },
            Entry { token: 9, bias: f32::NEG_INFINITY, rep: 1.0, sub: 0.0 },
        ];
        w.corrections(&entries).unwrap();
        let masks = [0b1010u32, 7, 1, 0xffff_ffff];
        w.masks(&masks, 2).unwrap();
        let params = SampleParams { temperature: 1.0, top_k: 3, ..SampleParams::GREEDY };
        w.samplers([params.row(10).with_entries(0, 2), params.row(11).with_mask(1)]).unwrap();
        w.commits(&[]).unwrap();
        let used = w.finish().unwrap();
        assert_eq!(used % 16, 0);
        let record = RecordView::new(&buf[..used]).unwrap();
        assert_eq!(record.corrections(), 2);
        assert_eq!((record.correction(0), record.correction(1)), (entries[0], entries[1]));
        assert_eq!(record.masks(), (2, 2));
        assert_eq!((record.mask_word(0, 1), record.mask_word(1, 0)), (7, 1));
        assert_eq!(record.sampler(0).entries(), (0, 2));
        assert_eq!(record.sampler(1).mask(), Some(1));
        // The sections are where the device finds them.
        let at = record.corrections_offset() as usize;
        assert_eq!(buf[at..at + Entry::BYTES], entries[0].to_bytes());
        let at = record.masks_offset() as usize;
        assert_eq!(u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()), 0b1010);
        // A record cut short of them is refused.
        assert_eq!(RecordView::new(&buf[..used - 16]).unwrap_err(), RecordError::Truncated);
    }
}
