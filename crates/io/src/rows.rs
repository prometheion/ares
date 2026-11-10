//! Reading rows of a huge table.
//!
//! Some models carry hashed-embedding tables of hundreds of millions of rows
//! of a few dozen bytes (28.8 GB for `qwen4exp`). A token needs a handful of
//! rows, and which ones is known on the host as soon as the token is known,
//! so the reads can start early and be collected before the layer that needs
//! them. Three parts make that cheap:
//!
//! - [`RowTable`] knows where rows are and turns a list of row ids into as
//!   few reads as the blocks allow: rows that share a block are read once,
//!   adjacent blocks are one read, and the reads come out sorted by offset.
//! - [`RowCache`] keeps recently used rows in a fixed amount of memory so
//!   that the same rows are not read twice.
//! - [`RowReader`] ties them to a [`Reader`]: it asks the cache first, reads
//!   the rest, and gives back the rows densely packed in the order asked.

use std::collections::HashMap;
use std::ops::Range;

use ares_gguf::TensorInfo;

use crate::buffer::Buffer;
use crate::error::{Error, LayoutError};
use crate::file::FileId;
use crate::reader::Reader;
use crate::request::{Class, Completion, Key, Read};
use crate::sched::IdMap;

/// The default cap on the length of one merged read.
const MAX_READ: usize = 256 * 1024;

/// A two-dimensional tensor stored row after row.
#[derive(Clone, Debug)]
pub struct RowTable {
    file: FileId,
    offset: u64,
    row_bytes: usize,
    n_rows: u64,
    block: u64,
    max_read: usize,
}

impl RowTable {
    /// The table in `info`, read in blocks of `block` bytes (the block size of
    /// its file; see [`RowTable::for_reader`]).
    ///
    /// # Errors
    ///
    /// The tensor is not two-dimensional, or its size does not fit.
    pub fn new(info: &TensorInfo, block: u64) -> Result<Self, LayoutError> {
        let dims = info.dims.as_slice();
        if dims.len() != 2 {
            return Err(LayoutError::NotATable { name: info.name.clone(), dims: dims.len() });
        }
        let n_rows = dims[1];
        let row_bytes = info.ty.bytes_for(dims[0]).and_then(|b| usize::try_from(b).ok());
        let too_large = || LayoutError::TableTooLarge { rows: n_rows, row_bytes: 0 };
        let row_bytes = row_bytes.ok_or_else(too_large)?;
        if (row_bytes as u64).checked_mul(n_rows).is_none() || row_bytes == 0 {
            return Err(LayoutError::TableTooLarge { rows: n_rows, row_bytes: row_bytes as u64 });
        }
        Ok(Self {
            file: FileId(info.part),
            offset: info.offset,
            row_bytes,
            n_rows,
            block: block.max(1),
            max_read: MAX_READ,
        })
    }

    /// The table in `info`, with the block size of the file `reader` reads it
    /// from.
    pub fn for_reader(reader: &Reader, info: &TensorInfo) -> Result<Self, LayoutError> {
        let file =
            reader.files().get(usize::from(info.part)).ok_or_else(|| LayoutError::NotInReader {
                name: info.name.clone(),
                part: info.part,
                files: reader.files().len(),
            })?;
        Self::new(info, file.block)
    }

    /// Caps the length of a merged read (default 256 KiB). A row is never
    /// split, so a row longer than the cap is read whole.
    pub fn with_max_read(mut self, max_read: usize) -> Self {
        self.max_read = max_read;
        self
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn n_rows(&self) -> u64 {
        self.n_rows
    }

    pub fn file(&self) -> FileId {
        self.file
    }

    /// Plans the reads for `rows` into `plan`, replacing its contents.
    ///
    /// Rows that lie in the same block (or the same row asked twice) share
    /// one read; blocks that touch are one read, up to the length cap; the
    /// reads are sorted by file offset, which is the order a drive prefers.
    ///
    /// # Errors
    ///
    /// A row id that is not in the table.
    pub fn plan(&self, rows: &[u64], plan: &mut RowPlan) -> Result<(), LayoutError> {
        plan.reads.clear();
        plan.slots.clear();
        plan.order.clear();
        if let Some(row) = rows.iter().copied().find(|r| *r >= self.n_rows) {
            return Err(LayoutError::RowOutOfRange { row, rows: self.n_rows });
        }
        plan.slots.resize(rows.len(), RowSlot::default());
        plan.order.extend(0..rows.len() as u32);
        plan.order.sort_unstable_by_key(|i| rows[*i as usize]);

        let row_bytes = self.row_bytes as u64;
        let mut open: Option<Open> = None;
        for (pos, &i) in plan.order.iter().enumerate() {
            let start = self.offset + rows[i as usize] * row_bytes;
            let end = start + row_bytes;
            let (first_block, last_block) = (start / self.block, (end - 1) / self.block);
            let extends = open.as_ref().is_some_and(|o| {
                first_block <= o.last_block + 1 && (end - o.start) as usize <= self.max_read
            });
            if !extends {
                if let Some(o) = open.take() {
                    plan.reads.push(o.finish(self.file, pos));
                }
                open = Some(Open { start, end, last_block, first_row: pos });
            }
            let o = open.as_mut().expect("opened above");
            o.end = end;
            o.last_block = last_block;
            plan.slots[i as usize] =
                RowSlot { read: plan.reads.len() as u32, at: (start - o.start) as u32 };
        }
        if let Some(o) = open {
            plan.reads.push(o.finish(self.file, plan.order.len()));
        }
        Ok(())
    }
}

struct Open {
    start: u64,
    end: u64,
    last_block: u64,
    /// Position in `RowPlan::order` of the read's first row.
    first_row: usize,
}

impl Open {
    fn finish(self, file: FileId, end_row: usize) -> PlannedRead {
        PlannedRead {
            file,
            offset: self.start,
            len: (self.end - self.start) as usize,
            first_row: self.first_row as u32,
            n_rows: (end_row - self.first_row) as u32,
        }
    }
}

/// One read of a plan: the range from the start of its first row to the end
/// of its last. [`Reader`] widens it to blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedRead {
    pub file: FileId,
    pub offset: u64,
    pub len: usize,
    first_row: u32,
    n_rows: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct RowSlot {
    read: u32,
    /// Where the row starts in the data of its read.
    at: u32,
}

/// The reads for a list of rows, and where each row is in their data.
/// Reusable: planning again replaces the contents without allocating once the
/// buffers have grown.
#[derive(Clone, Debug, Default)]
pub struct RowPlan {
    reads: Vec<PlannedRead>,
    /// Per requested row, in request order.
    slots: Vec<RowSlot>,
    /// Request indices sorted by row id; the rows of a read are contiguous.
    order: Vec<u32>,
}

impl RowPlan {
    /// The reads to make, sorted by offset.
    pub fn reads(&self) -> &[PlannedRead] {
        &self.reads
    }

    /// The number of rows planned.
    pub fn n_rows(&self) -> usize {
        self.slots.len()
    }

    /// Where requested row `i` is: the index of its read and its offset in
    /// that read's data (the wanted bytes, as [`Completion::data`] gives
    /// them).
    pub fn locate(&self, i: usize) -> (usize, usize) {
        let slot = self.slots[i];
        (slot.read as usize, slot.at as usize)
    }

    /// The bytes of requested row `i` in the data of its read.
    pub fn row_in<'a>(&self, i: usize, row_bytes: usize, data: &'a [u8]) -> &'a [u8] {
        let (_, at) = self.locate(i);
        &data[at..at + row_bytes]
    }

    /// The requested rows (by request index) that read `read` serves.
    pub fn rows_of(&self, read: usize) -> &[u32] {
        let r = &self.reads[read];
        &self.order[r.first_row as usize..(r.first_row + r.n_rows) as usize]
    }
}

/// What the cache has done.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// Memory the cache needs per row on top of the row itself: the row id, the
/// reference bit and the index entry.
const CACHE_ROW_OVERHEAD: usize = 8 + 1 + 16;

/// A cache of rows with a byte budget and CLOCK replacement.
///
/// **Policy.** The slots form a ring with a hand. A hit sets the slot's
/// reference bit. To make room the hand moves on, clearing set bits, and
/// evicts the first slot whose bit is clear. A new row enters with its bit
/// clear, so a row read once and never used again is the first to go, and a
/// row used since the hand last passed survives one more revolution. This
/// approximates least-recently-used for the cost of one bit per row and no
/// list to maintain on a hit.
///
/// **Budget.** All memory is reserved when the cache is created: the budget
/// divided by the row size plus 25 bytes of bookkeeping per row decides the
/// number of slots, and nothing is allocated afterwards.
#[derive(Debug)]
pub struct RowCache {
    row_bytes: usize,
    capacity: usize,
    index: IdMap<u32>,
    rows: Vec<u64>,
    referenced: Vec<bool>,
    data: Vec<u8>,
    hand: usize,
    stats: CacheStats,
}

impl RowCache {
    /// A cache for rows of `row_bytes` bytes within `budget` bytes of memory.
    /// A budget too small for one row gives a cache that holds nothing.
    pub fn new(row_bytes: usize, budget: usize) -> Self {
        let capacity = budget / (row_bytes + CACHE_ROW_OVERHEAD);
        Self {
            row_bytes,
            capacity,
            index: IdMap::with_capacity_and_hasher(capacity, Default::default()),
            rows: Vec::with_capacity(capacity),
            referenced: Vec::with_capacity(capacity),
            data: vec![0; capacity * row_bytes],
            hand: 0,
            stats: CacheStats::default(),
        }
    }

    /// The most rows the cache holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// The row, if it is cached; counts a hit or a miss and marks the row as
    /// used.
    pub fn get(&mut self, row: u64) -> Option<&[u8]> {
        match self.index.get(&row) {
            Some(&slot) => {
                self.stats.hits += 1;
                self.referenced[slot as usize] = true;
                Some(self.slot(slot as usize))
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Whether the row is cached, without counting or marking.
    pub fn contains(&self, row: u64) -> bool {
        self.index.contains_key(&row)
    }

    /// Stores a row, evicting by the CLOCK rule when the cache is full. A row
    /// that is cached already is left as it is.
    ///
    /// # Panics
    ///
    /// Panics when `bytes` is not `row_bytes` long.
    pub fn insert(&mut self, row: u64, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.row_bytes, "a cached row has the table's row size");
        if self.capacity == 0 || self.index.contains_key(&row) {
            return;
        }
        let slot = if self.rows.len() < self.capacity {
            self.rows.push(row);
            self.referenced.push(false);
            self.rows.len() - 1
        } else {
            while self.referenced[self.hand] {
                self.referenced[self.hand] = false;
                self.hand = (self.hand + 1) % self.capacity;
            }
            let slot = self.hand;
            self.hand = (self.hand + 1) % self.capacity;
            self.index.remove(&self.rows[slot]);
            self.stats.evictions += 1;
            self.rows[slot] = row;
            slot
        };
        self.index.insert(row, slot as u32);
        self.data[slot * self.row_bytes..][..self.row_bytes].copy_from_slice(bytes);
    }

    fn slot(&self, slot: usize) -> &[u8] {
        &self.data[slot * self.row_bytes..][..self.row_bytes]
    }
}

/// One gather of rows in progress: the rows asked for, the reads that are
/// outstanding for them, and the rows as they arrive. Create one empty with
/// [`RowGather::new`] and reuse it for every token.
#[derive(Debug, Default)]
pub struct RowGather {
    key_base: Key,
    row_bytes: usize,
    n_rows: usize,
    pending: usize,
    error: Option<Error>,
    /// Row ids of the rows that were not in the cache, in request order, and
    /// their positions among the requested rows.
    miss_rows: Vec<u64>,
    miss_pos: Vec<u32>,
    plan: RowPlan,
    /// The requested rows, packed in request order.
    out: Vec<u8>,
}

impl RowGather {
    pub fn new() -> Self {
        Self::default()
    }

    /// The keys of the reads of this gather: `key_base` up to `key_base` plus
    /// the number of reads. Completions with other keys belong to someone
    /// else.
    pub fn keys(&self) -> Range<Key> {
        self.key_base..self.key_base + self.plan.reads.len() as u64
    }

    /// Reads still outstanding.
    pub fn pending(&self) -> usize {
        self.pending
    }

    /// Whether every row has arrived (or the gather has failed and its
    /// outstanding reads are back).
    pub fn is_done(&self) -> bool {
        self.pending == 0
    }

    /// `Ok` when all rows arrived; otherwise the first error among the reads.
    pub fn result(&self) -> Result<(), &Error> {
        match &self.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// The number of rows asked for.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Requested row `i`. Only meaningful once the gather is done and
    /// succeeded.
    pub fn row(&self, i: usize) -> &[u8] {
        &self.out[i * self.row_bytes..][..self.row_bytes]
    }

    /// All requested rows, packed in request order: `n_rows * row_bytes`
    /// bytes, ready to be uploaded as they are.
    pub fn rows(&self) -> &[u8] {
        &self.out
    }
}

/// Reads rows of one table through a [`Reader`], with an optional cache.
///
/// It is single-threaded state like the rest of the step loop: it owns the
/// cache and a pool of read buffers (grown on demand, reused for ever
/// after), and several [`RowGather`]s can be in flight at once, for example
/// the rows of the next chunk of a prompt while the current one is consumed.
/// The caller polls the reader and hands completions to
/// [`RowReader::accept`]; completions that are not for a gather come back
/// out of it.
#[derive(Debug)]
pub struct RowReader {
    table: RowTable,
    cache: Option<RowCache>,
    alignment: usize,
    /// Free buffers by capacity (a power of two).
    pool: HashMap<usize, Vec<Buffer>>,
}

impl RowReader {
    /// A reader of `table` through `reader`, with a cache of `cache_budget`
    /// bytes if it is not zero.
    pub fn new(reader: &Reader, table: RowTable, cache_budget: usize) -> Self {
        let cache = (cache_budget > 0).then(|| RowCache::new(table.row_bytes(), cache_budget));
        Self { table, cache, alignment: reader.buffer_alignment(), pool: HashMap::new() }
    }

    pub fn table(&self) -> &RowTable {
        &self.table
    }

    pub fn cache(&self) -> Option<&RowCache> {
        self.cache.as_ref()
    }

    /// Starts the gather of `rows` into `gather`: rows in the cache are
    /// copied at once, the others are read with the keys `key_base`,
    /// `key_base + 1`, and so on (see [`RowGather::keys`]; the caller makes
    /// sure they are not in use).
    ///
    /// # Errors
    ///
    /// A row outside the table, or a read the reader refused. In that case
    /// the reads already queued are cancelled and their completions arrive as
    /// usual; the gather is empty.
    pub fn start(
        &mut self,
        reader: &Reader,
        rows: &[u64],
        key_base: Key,
        class: Class,
        gather: &mut RowGather,
    ) -> Result<(), Error> {
        let row_bytes = self.table.row_bytes();
        gather.key_base = key_base;
        gather.row_bytes = row_bytes;
        gather.n_rows = rows.len();
        gather.pending = 0;
        gather.error = None;
        gather.miss_rows.clear();
        gather.miss_pos.clear();
        gather.out.clear();
        gather.out.resize(rows.len() * row_bytes, 0);
        gather.plan.reads.clear();

        for (i, &row) in rows.iter().enumerate() {
            let hit = self.cache.as_mut().and_then(|cache| cache.get(row));
            match hit {
                Some(bytes) => gather.out[i * row_bytes..][..row_bytes].copy_from_slice(bytes),
                None => {
                    gather.miss_rows.push(row);
                    gather.miss_pos.push(i as u32);
                }
            }
        }
        self.table.plan(&gather.miss_rows, &mut gather.plan)?;

        for (r, read) in gather.plan.reads.iter().enumerate() {
            let span = reader.span(read.file, read.offset, read.len)?;
            let key = key_base + r as u64;
            let buffer = self.take_buffer(span.len);
            let submitted = reader.submit(Read {
                key,
                file: read.file,
                offset: read.offset,
                len: read.len,
                class,
                buffer,
            });
            if let Err(rejected) = submitted {
                self.give_back(rejected.read.buffer);
                for key in key_base..key {
                    let _ = reader.cancel(key);
                }
                gather.plan.reads.clear();
                gather.out.clear();
                return Err(rejected.error);
            }
            gather.pending += 1;
        }
        Ok(())
    }

    /// Hands a completion to the gather it belongs to. The rows of a
    /// successful read are copied to their places (and into the cache), and
    /// the buffer goes back to the pool; `None` is returned. A completion
    /// that is not one of the gather's is given back unchanged, for whoever
    /// else is waiting for reads.
    pub fn accept(&mut self, gather: &mut RowGather, completion: Completion) -> Option<Completion> {
        let keys = gather.keys();
        if gather.pending == 0 || !keys.contains(&completion.key) {
            return Some(completion);
        }
        let read = (completion.key - gather.key_base) as usize;
        gather.pending -= 1;
        let row_bytes = gather.row_bytes;
        if completion.result.is_ok() {
            let data = completion.data();
            for &i in gather.plan.rows_of(read) {
                let bytes = gather.plan.row_in(i as usize, row_bytes, data);
                let pos = gather.miss_pos[i as usize] as usize;
                gather.out[pos * row_bytes..][..row_bytes].copy_from_slice(bytes);
                if let Some(cache) = &mut self.cache {
                    cache.insert(gather.miss_rows[i as usize], bytes);
                }
            }
        } else if gather.error.is_none() {
            gather.error = completion.result.err();
        }
        self.give_back(completion.buffer);
        None
    }

    fn take_buffer(&mut self, len: usize) -> Buffer {
        let class = len.max(1).next_power_of_two();
        match self.pool.get_mut(&class).and_then(Vec::pop) {
            Some(buffer) => buffer,
            None => Buffer::aligned(class, self.alignment),
        }
    }

    fn give_back(&mut self, buffer: Buffer) {
        self.pool.entry(buffer.len()).or_default().push(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ares_core::GgmlType;
    use ares_gguf::Dims;

    /// A table of `n_rows` rows of 90 bytes (45 `f16` values) at file offset
    /// 1000 (as GGUF data would sit: aligned to 32, not to a block).
    fn table(n_rows: u64, block: u64) -> RowTable {
        let info = TensorInfo {
            name: "table".into(),
            dims: Dims::new(&[45, n_rows]).unwrap(),
            ty: GgmlType::F16,
            part: 0,
            offset: 1024,
            size: 90 * n_rows,
        };
        RowTable::new(&info, block).unwrap()
    }

    /// (offset, len) of each planned read.
    fn reads(plan: &RowPlan) -> Vec<(u64, usize)> {
        plan.reads().iter().map(|r| (r.offset, r.len)).collect()
    }

    #[test]
    fn rows_in_one_block_are_one_read() {
        let t = table(1000, 4096);
        let mut plan = RowPlan::default();
        // Rows 3, 4 and 9 all lie in block 0 of the file (offsets 1294..1914).
        t.plan(&[9, 3, 4], &mut plan).unwrap();
        assert_eq!(reads(&plan), [(1024 + 3 * 90, 7 * 90)], "from row 3's start to row 9's end");
        for (i, row) in [9u64, 3, 4].iter().enumerate() {
            let (read, at) = plan.locate(i);
            assert_eq!(read, 0);
            assert_eq!(at as u64, (row - 3) * 90);
        }
    }

    #[test]
    fn duplicate_rows_share_a_read_and_a_place() {
        let t = table(1000, 4096);
        let mut plan = RowPlan::default();
        t.plan(&[7, 7, 7], &mut plan).unwrap();
        assert_eq!(reads(&plan), [(1024 + 7 * 90, 90)]);
        assert!((0..3).all(|i| plan.locate(i) == (0, 0)));
        assert_eq!(plan.rows_of(0).len(), 3);
    }

    #[test]
    fn rows_in_touching_blocks_merge_and_far_ones_do_not() {
        let t = table(1000, 4096);
        let mut plan = RowPlan::default();
        // Rows in blocks 0, 1 and 5 of the file (the table starts at 1024).
        let block_of = |row: u64| (1024 + row * 90) / 4096;
        let in_block = |b: u64| (0..1000).find(|r| block_of(*r) == b).unwrap();
        let (a, b, c) = (in_block(0), in_block(1), in_block(5));
        t.plan(&[c, a, b], &mut plan).unwrap();
        let got = reads(&plan);
        assert_eq!(got.len(), 2, "blocks 0 and 1 touch, block 5 is apart: {got:?}");
        assert!(got[0].0 < got[1].0, "sorted by offset");
        // The first read covers a..=b; the request order maps back correctly.
        assert_eq!(plan.locate(1).0, 0);
        assert_eq!(plan.locate(2).0, 0);
        assert_eq!(plan.locate(0).0, 1);
        assert_eq!(plan.rows_of(1), [0]);
    }

    #[test]
    fn a_row_across_a_block_boundary_takes_both_blocks() {
        let t = table(1000, 4096);
        let mut plan = RowPlan::default();
        // Find a row that straddles a boundary of 4096.
        let row = (0..1000u64)
            .find(|r| {
                let s = 1024 + r * 90;
                s / 4096 != (s + 89) / 4096
            })
            .unwrap();
        t.plan(&[row], &mut plan).unwrap();
        let (offset, len) = reads(&plan)[0];
        assert_eq!((offset, len), (1024 + row * 90, 90));
        // Widened to blocks, the read covers 8192 bytes: the whole point of
        // keeping the row in one read.
        let span = crate::Span::new(offset, len, 4096).unwrap();
        assert_eq!(span.len, 8192);
    }

    #[test]
    fn the_cap_splits_long_runs_of_rows() {
        let t = table(1000, 512).with_max_read(1000);
        let mut plan = RowPlan::default();
        let rows: Vec<u64> = (0..100).collect();
        t.plan(&rows, &mut plan).unwrap();
        let got = reads(&plan);
        assert!(got.len() >= 9 && got.iter().all(|(_, len)| *len <= 1000), "{got:?}");
        // The reads tile the rows without gaps or overlap.
        assert_eq!(got.iter().map(|(_, l)| *l).sum::<usize>(), 100 * 90);
        // A cap below the row size still reads whole rows.
        let t = table(10, 512).with_max_read(10);
        t.plan(&[1, 2], &mut plan).unwrap();
        assert_eq!(reads(&plan), [(1024 + 90, 90), (1024 + 180, 90)]);
    }

    #[test]
    fn planning_checks_row_ids_and_reuses_the_plan() {
        let t = table(10, 4096);
        let mut plan = RowPlan::default();
        assert!(matches!(
            t.plan(&[3, 10], &mut plan),
            Err(LayoutError::RowOutOfRange { row: 10, rows: 10 })
        ));
        t.plan(&[], &mut plan).unwrap();
        assert!(plan.reads().is_empty() && plan.n_rows() == 0);
        t.plan(&[2, 1], &mut plan).unwrap();
        assert_eq!(plan.n_rows(), 2);
    }

    #[test]
    fn only_two_dimensional_tensors_are_tables() {
        let info = TensorInfo {
            name: "t".into(),
            dims: Dims::new(&[45, 3, 2]).unwrap(),
            ty: GgmlType::F16,
            part: 0,
            offset: 0,
            size: 540,
        };
        assert!(matches!(RowTable::new(&info, 512), Err(LayoutError::NotATable { dims: 3, .. })));
    }

    fn row(n: u8) -> [u8; 10] {
        [n; 10]
    }

    #[test]
    fn the_cache_keeps_to_its_budget() {
        let cache = RowCache::new(10, 1000);
        assert_eq!(cache.capacity(), 1000 / (10 + CACHE_ROW_OVERHEAD));
        assert!(cache.capacity() * (10 + CACHE_ROW_OVERHEAD) <= 1000);
        let mut none = RowCache::new(10, 20);
        assert_eq!(none.capacity(), 0);
        none.insert(1, &row(1));
        assert!(none.get(1).is_none() && none.is_empty());
    }

    #[test]
    fn the_cache_returns_what_was_stored_and_counts() {
        let mut cache = RowCache::new(10, 350); // 10 slots
        assert_eq!(cache.capacity(), 10);
        assert!(cache.get(5).is_none());
        cache.insert(5, &row(5));
        assert_eq!(cache.get(5), Some(&row(5)[..]));
        cache.insert(5, &row(99)); // already there: left alone
        assert_eq!(cache.get(5), Some(&row(5)[..]));
        assert_eq!(cache.stats(), CacheStats { hits: 2, misses: 1, evictions: 0 });
        assert!(cache.contains(5) && !cache.contains(6));
        assert_eq!(cache.stats().hits, 2, "contains does not count");
    }

    #[test]
    fn clock_evicts_rows_that_were_not_used_since_the_hand_passed() {
        let mut cache = RowCache::new(10, 350); // 10 slots
        for r in 0..10u64 {
            cache.insert(r, &row(r as u8));
        }
        // Use rows 0..5; rows 5..10 were never touched.
        for r in 0..5 {
            assert!(cache.get(r).is_some());
        }
        for r in 10..15u64 {
            cache.insert(r, &row(r as u8));
        }
        for r in 0..5 {
            assert!(cache.contains(r), "row {r} was used, so it survives");
        }
        for r in 5..10 {
            assert!(!cache.contains(r), "row {r} was never used, so it went");
        }
        for r in 10..15 {
            assert_eq!(cache.get(r), Some(&row(r as u8)[..]));
        }
        assert_eq!(cache.stats().evictions, 5);
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn clock_gives_a_second_chance_and_then_evicts() {
        let mut cache = RowCache::new(10, 105); // 3 slots
        assert_eq!(cache.capacity(), 3);
        for r in 0..3u64 {
            cache.insert(r, &row(r as u8));
            assert!(cache.get(r).is_some()); // every row is marked as used
        }
        // All marked: the hand clears every mark on its way round and then
        // evicts the slot it started at (row 0).
        cache.insert(3, &row(3));
        assert!(!cache.contains(0) && cache.contains(1) && cache.contains(2) && cache.contains(3));
        // Row 1 and 2 lost their marks, row 3 never had one: the hand is at
        // row 1's slot, so the next insert evicts row 1, then row 2.
        cache.insert(4, &row(4));
        assert!(!cache.contains(1) && cache.contains(2));
        cache.insert(5, &row(5));
        assert!(!cache.contains(2) && cache.contains(3) && cache.contains(4) && cache.contains(5));
        assert_eq!(cache.stats().evictions, 3);
    }
}
