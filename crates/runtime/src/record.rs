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
//! | commit record | 8 + 8 each | a kernel header and the segments of the last step whose rows were drafts, with the rows that were accepted |
//! | updates | 3 each | sequence slot, first page-table entry, number of entries |
//! | entries | 1 each | page-table entries, for the updates in order |
//!
//! A segment is **direct** when all of its rows are certain to be accepted (a
//! slice of a prompt, a generation row without drafts): the recurrent kernels
//! then write the sequence's state themselves. The rows of any other segment
//! leave recurrent state untouched, and the next step carries, in its commit
//! record, how many of them were accepted. The commit record is itself in the
//! kernels' layout, so its device address is the `step` argument of the commit
//! kernels.
//!
//! The *context* is a number the host derives from the sequence's tokens up to
//! and including the row's token. A real model has no use for it (attention
//! reads the sequence's pages); the synthetic model has no attention kernel,
//! so this is how a row's result comes to depend on its whole sequence.

use ares_cuda::{Segment, StepDims};

const MAGIC: u32 = 0x5354_4550;
/// Bytes of the kernels' header.
const KERNEL_HEADER: usize = 32;
/// Bytes of both headers.
const HEADERS: usize = KERNEL_HEADER + 64;
const SEG_BYTES: usize = 32;
const FLAG_DIRECT: u32 = 1;

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

/// The bytes the record can take at most, for buffer sizes.
pub fn max_bytes(max_rows: u32, max_slots: u32) -> usize {
    let (rows, slots) = (max_rows as usize, max_slots as usize);
    // A segment and a commit per sequence at worst, a run per row; a step
    // creates at most one page per row and re-dirties at most one entry per
    // sequence, and a sequence has at most one update.
    let segments = slots.min(rows.max(1));
    let bytes = HEADERS
        + SEG_BYTES * segments
        + rows * (4 + 16 + 8 + 4 + 8)
        + KERNEL_HEADER
        + SEG_BYTES * slots
        + slots * 12
        + (rows + 2 * slots) * 4
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
    header: [u32; 24],
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
        let mut header = [0u32; 24];
        header[0] = n_rows;
        header[1] = n_segs;
        header[4] = off_segs as u32;
        header[5] = off_tokens as u32;
        header[6] = off_pos as u32;
        header[7] = end as u32;
        header[H_MAGIC] = MAGIC;
        header[H_STEP] = step as u32;
        header[H_OFF_EXTRA] = off_extra as u32;
        for h in [H_OFF_RUNS, H_OFF_COMMIT, H_OFF_UPDATES, H_OFF_ENTRIES] {
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
        self.end = align(at);
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
    header: [u32; 24],
}

impl<'a> RecordView<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, RecordError> {
        if bytes.len() < HEADERS {
            return Err(RecordError::Short);
        }
        let mut header = [0u32; 24];
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
            header[H_OFF_COMMIT] as usize
                + if header[H_COMMITS] > 0 {
                    KERNEL_HEADER + header[H_COMMITS] as usize * SEG_BYTES
                } else {
                    0
                },
            header[H_OFF_UPDATES] as usize + header[H_UPDATES] as usize * 12,
            header[H_OFF_ENTRIES] as usize + header[H_ENTRIES] as usize * 4,
        ];
        if ends.iter().any(|end| *end > bytes.len()) {
            return Err(RecordError::Truncated);
        }
        Ok(view)
    }

    fn word(&self, byte: usize) -> u32 {
        u32::from_le_bytes(self.bytes[byte..byte + 4].try_into().expect("4 bytes"))
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
        let bytes = max_bytes(64, 256);
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
        assert!(w.finish().unwrap() <= bytes);
    }
}
