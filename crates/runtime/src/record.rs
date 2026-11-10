//! The step record: everything that varies from step to step, in one buffer.
//!
//! Kernels do not take the rows of a step as arguments. The conductor writes
//! them into one page-locked buffer, each device gets that buffer with one
//! copy, and launches carry only pointers into it and sizes. So a step of any
//! shape costs the same number of copies and the same launches.
//!
//! Layout, little-endian `u32` words, sections one after the other so that
//! only the used bytes are copied:
//!
//! | section | words | content |
//! | --- | --- | --- |
//! | header | 8 | magic, step number, then the number of rows, runs, updates, entries and picked rows |
//! | rows | 4 each | sequence slot, position, token, context |
//! | runs | 2 each | first row, number of rows: the rows whose logits are wanted, in order |
//! | updates | 3 each | sequence slot, first page-table entry, number of entries |
//! | entries | 1 each | page numbers, for the updates in order |
//!
//! The *context* is a number the host derives from the sequence's tokens up to
//! and including the row's token. A real model has no use for it (attention
//! reads the sequence's pages); the synthetic model has no attention kernel,
//! so this is how a row's result comes to depend on its whole sequence.

const MAGIC: u32 = 0x5354_4550;
/// Words in the header.
pub const HEADER_WORDS: usize = 8;
/// Words per row.
pub const ROW_WORDS: usize = 4;
const RUN_WORDS: usize = 2;
const UPDATE_WORDS: usize = 3;

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

/// The bytes the record can take at most, for buffer sizes.
pub fn max_bytes(max_rows: u32, max_slots: u32) -> usize {
    // One run per row at worst; a step creates at most one page per row and
    // re-dirties at most one entry per sequence, and a sequence has at most
    // one update.
    let words = HEADER_WORDS
        + max_rows as usize * (ROW_WORDS + RUN_WORDS + 1)
        + max_slots as usize * (UPDATE_WORDS + 2);
    (words * 4).next_multiple_of(256)
}

/// The record does not fit its buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the step record does not fit its {capacity}-byte buffer")]
pub struct RecordFull {
    pub capacity: usize,
}

/// Writes a record into a buffer. Sections must be written in order: all
/// rows, then all runs, then all updates, then all entries.
#[derive(Debug)]
pub struct RecordWriter<'a> {
    buf: &'a mut [u8],
    words: usize,
    counts: [u32; 4],
    /// The section being written: 0 rows, 1 runs, 2 updates, 3 entries.
    section: usize,
    picked: u32,
    step: u64,
}

impl<'a> RecordWriter<'a> {
    pub fn new(buf: &'a mut [u8], step: u64) -> Self {
        Self { buf, words: HEADER_WORDS, counts: [0; 4], section: 0, picked: 0, step }
    }

    fn put(&mut self, section: usize, words: &[u32]) -> Result<(), RecordFull> {
        debug_assert!(section >= self.section, "record sections are written in order");
        self.section = section;
        let end = (self.words + words.len()) * 4;
        let Some(dst) = self.buf.get_mut(self.words * 4..end) else {
            return Err(RecordFull { capacity: self.buf.len() });
        };
        for (chunk, word) in dst.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        self.words += words.len();
        self.counts[section] += 1;
        Ok(())
    }

    pub fn row(&mut self, row: Row) -> Result<(), RecordFull> {
        self.put(0, &[row.slot, row.position, row.token, row.context])
    }

    /// Rows `first..first + count` need their logits.
    pub fn run(&mut self, first: u32, count: u32) -> Result<(), RecordFull> {
        self.picked += count;
        self.put(1, &[first, count])
    }

    pub fn update(&mut self, update: TableUpdate) -> Result<(), RecordFull> {
        self.put(2, &[update.slot, update.first, update.count])
    }

    pub fn entry(&mut self, page: u32) -> Result<(), RecordFull> {
        self.put(3, &[page])
    }

    /// Writes the header and returns the bytes used, rounded up to 16 so the
    /// copy is a whole number of the device's transfer units.
    pub fn finish(self) -> Result<usize, RecordFull> {
        let used = (self.words * 4).next_multiple_of(16);
        let capacity = self.buf.len();
        let Some(dst) = self.buf.get_mut(..used) else {
            return Err(RecordFull { capacity });
        };
        let header = [
            MAGIC,
            self.step as u32,
            self.counts[0],
            self.counts[1],
            self.counts[2],
            self.counts[3],
            self.picked,
            0,
        ];
        for (chunk, word) in dst.as_chunks_mut::<4>().0.iter_mut().zip(header) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        // Padding words keep whatever the last step left there; nothing reads
        // them, because every section is counted in the header.
        Ok(used)
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
    counts: [u32; 4],
    picked: u32,
    step: u32,
}

impl<'a> RecordView<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, RecordError> {
        let word = |i: usize| -> Option<u32> {
            bytes.get(i * 4..i * 4 + 4).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        };
        let mut header = [0u32; HEADER_WORDS];
        for (i, slot) in header.iter_mut().enumerate() {
            *slot = word(i).ok_or(RecordError::Short)?;
        }
        if header[0] != MAGIC {
            return Err(RecordError::BadMagic);
        }
        let counts = [header[2], header[3], header[4], header[5]];
        let words = HEADER_WORDS
            + counts[0] as usize * ROW_WORDS
            + counts[1] as usize * RUN_WORDS
            + counts[2] as usize * UPDATE_WORDS
            + counts[3] as usize;
        if bytes.len() < words * 4 {
            return Err(RecordError::Truncated);
        }
        Ok(Self { bytes, counts, picked: header[6], step: header[1] })
    }

    fn word(&self, i: usize) -> u32 {
        u32::from_le_bytes(self.bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"))
    }

    /// The low 32 bits of the step number.
    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn rows(&self) -> u32 {
        self.counts[0]
    }

    pub fn runs(&self) -> u32 {
        self.counts[1]
    }

    pub fn updates(&self) -> u32 {
        self.counts[2]
    }

    pub fn entries(&self) -> u32 {
        self.counts[3]
    }

    /// The number of rows whose logits are wanted (the sum of the runs).
    pub fn picked_rows(&self) -> u32 {
        self.picked
    }

    pub fn row(&self, i: u32) -> Row {
        assert!(i < self.counts[0], "row {i} of {}", self.counts[0]);
        let at = HEADER_WORDS + i as usize * ROW_WORDS;
        Row {
            slot: self.word(at),
            position: self.word(at + 1),
            token: self.word(at + 2),
            context: self.word(at + 3),
        }
    }

    /// `(first row, rows)` of run `i`.
    pub fn run(&self, i: u32) -> (u32, u32) {
        assert!(i < self.counts[1], "run {i} of {}", self.counts[1]);
        let at = HEADER_WORDS + self.counts[0] as usize * ROW_WORDS + i as usize * RUN_WORDS;
        (self.word(at), self.word(at + 1))
    }

    pub fn update(&self, i: u32) -> TableUpdate {
        assert!(i < self.counts[2], "update {i} of {}", self.counts[2]);
        let at = HEADER_WORDS
            + self.counts[0] as usize * ROW_WORDS
            + self.counts[1] as usize * RUN_WORDS
            + i as usize * UPDATE_WORDS;
        TableUpdate { slot: self.word(at), first: self.word(at + 1), count: self.word(at + 2) }
    }

    /// The word offset of the entries section, in 32-bit words from the start
    /// of the record: the device copy of an update's entries is there.
    pub fn entries_offset_words(&self) -> usize {
        HEADER_WORDS
            + self.counts[0] as usize * ROW_WORDS
            + self.counts[1] as usize * RUN_WORDS
            + self.counts[2] as usize * UPDATE_WORDS
    }

    pub fn entry(&self, i: u32) -> u32 {
        assert!(i < self.counts[3], "entry {i} of {}", self.counts[3]);
        self.word(self.entries_offset_words() + i as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Vec<u8>, usize) {
        let mut buf = vec![0xAAu8; 1024];
        let mut w = RecordWriter::new(&mut buf, 77);
        w.row(Row { slot: 3, position: 10, token: 65, context: 900 }).unwrap();
        w.row(Row { slot: 3, position: 11, token: 66, context: 901 }).unwrap();
        w.row(Row { slot: 5, position: 0, token: 67, context: 902 }).unwrap();
        w.run(0, 2).unwrap();
        w.run(2, 1).unwrap();
        w.update(TableUpdate { slot: 3, first: 1, count: 2 }).unwrap();
        w.entry(40).unwrap();
        w.entry(41).unwrap();
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
        assert_eq!(record.entries_offset_words(), HEADER_WORDS + 3 * 4 + 2 * 2 + 3);
    }

    #[test]
    fn a_record_that_does_not_fit_is_an_error_and_not_a_panic() {
        let mut buf = vec![0u8; HEADER_WORDS * 4 + 4 * 4];
        let mut w = RecordWriter::new(&mut buf, 1);
        w.row(Row { slot: 0, position: 0, token: 0, context: 0 }).unwrap();
        let full = w.row(Row { slot: 0, position: 1, token: 0, context: 0 }).unwrap_err();
        assert_eq!(full.capacity, buf.len());
    }

    #[test]
    fn bytes_that_are_not_a_record_are_refused() {
        assert_eq!(RecordView::new(&[0; 8]).unwrap_err(), RecordError::Short);
        assert_eq!(RecordView::new(&[0; 64]).unwrap_err(), RecordError::BadMagic);
        let (buf, used) = sample();
        assert_eq!(RecordView::new(&buf[..used - 16]).unwrap_err(), RecordError::Truncated);
    }

    #[test]
    fn the_buffer_size_covers_a_full_step() {
        // 64 rows of 64 sequences each with a new page, and 256 sequences with
        // a table update: the worst case the planner can build.
        let bytes = max_bytes(64, 256);
        let words =
            HEADER_WORDS + 64 * ROW_WORDS + 64 * RUN_WORDS + 256 * UPDATE_WORDS + 64 + 2 * 256;
        assert!(bytes >= words * 4);
    }
}
