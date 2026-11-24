//! What a device part does for the prefix cache, one duty at a time, on the
//! real Qwen3.5-0.8B on one card: no conductor, no sampling, bit for bit.
//!
//! Every comparison is between runs that read the rows in steps of the same
//! sizes, so that rounding (which depends on the size of a step) cannot
//! show: what differs is the cache's doing.
//!
//! - **A replay writes no cell.** Its rows are given other tokens than the
//!   ones that wrote the cells it reads; the pool must stay as it was. The
//!   same rows without the replay flag change it (the control).
//! - **A replay rebuilds the state.** Rows replayed over cells that exist
//!   leave the recurrent state the rows that wrote them left.
//! - **A saved state restores exactly.** A sequence that restores the state
//!   saved at a page boundary and reads the next rows next to the pages the
//!   first one wrote gets the logits and the state the first one got.
//! - **A page copy carries every block's cells.** A sequence that reads the
//!   copy of a page where the first one reads the page gets the same logits.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_0_8B`.

use std::ops::Range;
use std::path::PathBuf;

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, ENTRY_CELLS, Kernels, Precision, Segment, StepRecord, Stream, Wait,
};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{
    Options, PAGE_ENTRY, Part, PartSetup, Placement, StepEnv, page_bytes,
};
use ares_model::qwen35::{Config, Weights};
use ares_runtime::PinnedRam;
use ares_tokenizer::{EncodeOptions, Tokenizer};
use common::helpers::floats;

mod common;

const PAGE: u32 = ENTRY_CELLS;

fn model_path() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    match std::env::var_os("ARES_TEST_QWEN35_0_8B") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_0_8B to run this test");
            None
        }
    }
}

/// The part with its buffers, as the runtime would own them: two slots, a pool
/// of pages that the tests hand out, and room for two checkpoints.
struct Rig {
    stream: Stream,
    part: Part,
    record: DeviceBuffer,
    tables: DeviceBuffer,
    pages: DeviceBuffer,
    table_stride: u32,
    slots: u32,
    logits: usize,
}

/// A step's logits and the state of the slot after it.
struct Run {
    logits: Vec<f32>,
}

impl Rig {
    fn open(device: &Device, c: &Config, w: &Weights<'_>, placement: &Placement) -> Self {
        let options = Options {
            precision: Precision::Exact,
            max_rows: 128,
            logit_rows: 4,
            window_rows: 0,
            ..Options::default()
        };
        let (slots, table_stride, pages) = (2u32, 8u32, 16u32);
        let kernels = Kernels::load(device).unwrap();
        let mut ledger =
            Ledger::new(format!("CUDA{}", device.ordinal()), device.mem_info().unwrap().free);
        let ram = PinnedRam::new(1 << 30);
        let part = Part::open(
            c,
            w,
            placement,
            &options,
            PartSetup {
                index: 0,
                device,
                kernels: &kernels,
                ledger: &mut ledger,
                max_slots: slots,
                pages,
                table_stride,
                checkpoints: 2,
                ram: &ram,
                table: None,
            },
        )
        .unwrap();
        let pool = u64::from(pages) * page_bytes(c, placement, 0, &options);
        Self {
            stream: Stream::new(device, 0).unwrap(),
            part,
            record: DeviceBuffer::new(device, 256 * 1024).unwrap(),
            tables: DeviceBuffer::new(device, u64::from(slots * table_stride) * 4).unwrap(),
            pages: DeviceBuffer::new(device, pool).unwrap(),
            table_stride,
            slots,
            logits: c.n_vocab,
        }
    }

    /// Sets the pages of `slot`: entry `i` is page `pages[i]`.
    fn table(&mut self, slot: u32, pages: &[u32]) {
        assert!(slot < self.slots && pages.len() <= self.table_stride as usize);
        let entries: Vec<u32> = pages.iter().map(|p| p | PAGE_ENTRY).collect();
        let bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
        self.tables.write(u64::from(slot * self.table_stride) * 4, &bytes).unwrap();
        self.part.update_table(slot, 0, entries.iter().copied()).unwrap();
    }

    /// Reads `tokens` from `pos0` in `slot` as one step, whose last row's
    /// logits are wanted; the rows of `replay` (counted from the step's first)
    /// are a replay.
    fn step(&mut self, slot: u32, pos0: u32, tokens: &[u32], replay: bool) -> Run {
        let n = tokens.len() as u32;
        let segment = Segment { row0: 0, n_rows: n, slot, pos0: pos0 as i32, direct: true };
        let record = StepRecord::new(&[segment], tokens, None, &[n - 1]).unwrap();
        self.record.write(0, record.as_bytes()).unwrap();
        let rows: Vec<Range<u32>> =
            if replay { std::iter::once(0..n).collect() } else { Vec::new() };
        let segments = [segment];
        let env = StepEnv {
            stream: &self.stream,
            step: self.record.ptr(),
            dims: record.dims(),
            tokens,
            segments: &segments,
            tables: self.tables.ptr(),
            pages: self.pages.ptr(),
            exchange: None,
            replay: &rows,
        };
        let logits = self.part.enqueue(&env).unwrap();
        self.stream.synchronize().unwrap();
        let logits = logits.expect("the part has the head");
        let mut bytes = vec![0u8; self.logits * 4];
        self.part.scratch().read(self.part.logits_offset(), &mut bytes).unwrap();
        assert_eq!(logits.stride as usize, self.logits);
        Run { logits: floats(&bytes) }
    }

    /// The whole pool.
    fn pool(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.pages.len() as usize];
        self.pages.read(0, &mut bytes).unwrap();
        bytes
    }

    /// The recurrent state of `slot`.
    fn state(&self, slot: u32) -> Vec<u8> {
        let per = self.part.state_bytes_per_slot();
        let mut bytes = vec![0u8; per as usize];
        self.part.state().read(u64::from(slot) * per, &mut bytes).unwrap();
        bytes
    }

    fn save(&mut self, slot: u32, place: u32) {
        self.part.save_state(&self.stream, slot, place).unwrap();
        self.stream.synchronize().unwrap();
    }

    fn restore(&mut self, slot: u32, place: u32) {
        self.part.restore_state(&self.stream, slot, place).unwrap();
        self.stream.synchronize().unwrap();
    }

    fn copy_page(&mut self, from: u32, to: u32) {
        self.part.copy_page(&self.stream, self.pages.ptr(), from, to).unwrap();
        self.stream.synchronize().unwrap();
    }
}

/// The first position where two byte strings part, or `None`.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or((a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// A prompt of at least `n` tokens and the same text with the words reversed
/// (other tokens, as many).
fn tokens(tokenizer: &Tokenizer, n: usize) -> (Vec<u32>, Vec<u32>) {
    let words: Vec<String> =
        (0..n).map(|i| format!("word{} of the {}th page", i * 7 % 101, i)).collect();
    let text = words.join(" ");
    let other = words.iter().rev().cloned().collect::<Vec<_>>().join(" ");
    let encode =
        |t: &str| tokenizer.encode(t, EncodeOptions { add_special: true, parse_special: false });
    let (mut a, mut b) = (encode(&text), encode(&other));
    let len = a.len().min(b.len());
    assert!(len >= n, "{len} tokens");
    a.truncate(len);
    b.truncate(len);
    (a, b)
}

struct Setup {
    device: Device,
    config: Config,
    gguf: Gguf,
    ids: Vec<u32>,
    other: Vec<u32>,
}

fn setup() -> Option<Setup> {
    let path = model_path()?;
    let gguf = Gguf::open(&path).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let (ids, other) = tokens(&tokenizer, 200);
    Some(Setup { device: Device::open(0, Wait::Block).unwrap(), config, gguf, ids, other })
}

#[test]
fn a_replay_writes_no_cell_and_rebuilds_the_recurrent_state() {
    let Some(s) = setup() else { return };
    let weights = Weights::load(&s.gguf, &s.config).unwrap();
    let placement = Placement::single(&s.config);
    let mut rig = Rig::open(&s.device, &s.config, &weights, &placement);
    let rows = 2 * PAGE as usize;

    // Slot 0 reads two pages of text and writes their cells.
    rig.table(0, &[0, 1, 2, 3]);
    for (i, chunk) in s.ids[..rows].chunks(PAGE as usize).enumerate() {
        rig.step(0, i as u32 * PAGE, chunk, false);
    }
    let (pool, written) = (rig.pool(), rig.state(0));
    assert!(pool.iter().any(|b| *b != 0), "the rows wrote cells");

    // Slot 1 replays the rows that wrote the cells, over the same pages: it
    // gets the state they left, and the pool is as it was.
    rig.table(1, &[0, 1, 4, 5]);
    for (i, chunk) in s.ids[..rows].chunks(PAGE as usize).enumerate() {
        rig.step(1, i as u32 * PAGE, chunk, true);
    }
    let at = first_difference(&written, &rig.state(1));
    assert_eq!(at, None, "the replayed state parts from the written one at byte {at:?}");
    let at = first_difference(&pool, &rig.pool());
    assert_eq!(at, None, "a replay wrote into the pool at byte {at:?}");

    // Other tokens replayed over the cells write nothing either.
    for (i, chunk) in s.other[..rows].chunks(PAGE as usize).enumerate() {
        rig.step(1, i as u32 * PAGE, chunk, true);
    }
    let at = first_difference(&pool, &rig.pool());
    assert_eq!(at, None, "a replay of other tokens wrote into the pool at byte {at:?}");
    // The control: the same rows as rows that write change the pool.
    for (i, chunk) in s.other[..rows].chunks(PAGE as usize).enumerate() {
        rig.step(1, i as u32 * PAGE, chunk, false);
    }
    assert!(
        first_difference(&pool, &rig.pool()).is_some(),
        "rows of other tokens written to the same pages change them"
    );
    eprintln!(
        "RES a replay of {rows} rows left the pool of {} bytes as it was and the state of {} \
         bytes as the rows that wrote the cells left it",
        pool.len(),
        written.len()
    );
}

#[test]
fn a_state_saved_at_a_page_boundary_restores_into_another_slot_exactly() {
    let Some(s) = setup() else { return };
    let weights = Weights::load(&s.gguf, &s.config).unwrap();
    let placement = Placement::single(&s.config);
    let mut rig = Rig::open(&s.device, &s.config, &weights, &placement);
    let page = PAGE as usize;

    // Slot 0: two pages, a checkpoint, then a third page.
    rig.table(0, &[0, 1, 2, 3]);
    for (i, chunk) in s.ids[..2 * page].chunks(page).enumerate() {
        rig.step(0, i as u32 * PAGE, chunk, false);
    }
    rig.save(0, 1);
    let saved = rig.state(0);
    let next = &s.ids[2 * page..3 * page];
    let first = rig.step(0, 2 * PAGE, next, false);
    let first_state = rig.state(0);

    // Slot 1 shares the first two pages, restores the checkpoint (from a
    // place that is not the first, so that the place is checked) and reads
    // the same third page into a page of its own.
    rig.table(1, &[0, 1, 6, 7]);
    rig.restore(1, 1);
    let at = first_difference(&saved, &rig.state(1));
    assert_eq!(at, None, "the restored state parts from the saved one at byte {at:?}");
    let second = rig.step(1, 2 * PAGE, next, false);
    let at = first_difference(&bytes(&first.logits), &bytes(&second.logits));
    assert_eq!(at, None, "the logits part at byte {at:?}");
    let at = first_difference(&first_state, &rig.state(1));
    assert_eq!(at, None, "the states part at byte {at:?}");
    eprintln!(
        "RES a restored state gave the logits of {} values and the state of {} bytes bit for bit",
        first.logits.len(),
        first_state.len()
    );

    // The checkpoint is a copy: the state of slot 0 moving on did not touch it.
    rig.table(1, &[0, 1, 6, 7]);
    rig.restore(1, 1);
    let again = rig.step(1, 2 * PAGE, next, false);
    assert_eq!(first_difference(&bytes(&first.logits), &bytes(&again.logits)), None);
}

#[test]
fn a_copied_page_is_read_as_the_page_it_copies() {
    let Some(s) = setup() else { return };
    let weights = Weights::load(&s.gguf, &s.config).unwrap();
    let placement = Placement::single(&s.config);
    let mut rig = Rig::open(&s.device, &s.config, &weights, &placement);
    let page = PAGE as usize;

    // Slot 0 reads three pages, saving the state after two and after three.
    rig.table(0, &[0, 1, 2, 3]);
    for (i, chunk) in s.ids[..2 * page].chunks(page).enumerate() {
        rig.step(0, i as u32 * PAGE, chunk, false);
    }
    rig.step(0, 2 * PAGE, &s.ids[2 * page..3 * page], false);
    rig.save(0, 0);
    // The fourth page is read next to the three.
    let next = &s.ids[3 * page..4 * page];
    let want = rig.step(0, 3 * PAGE, next, false);

    // Slot 1 reads the third page from its copy and the first two from slot
    // 0's pages.
    rig.copy_page(2, 8);
    rig.table(1, &[0, 1, 8, 9]);
    rig.restore(1, 0);
    let got = rig.step(1, 3 * PAGE, next, false);
    let at = first_difference(&bytes(&want.logits), &bytes(&got.logits));
    assert_eq!(at, None, "the logits part at byte {at:?}");
    eprintln!(
        "RES the rows after a copied page gave the logits of the rows after the page, bit for bit"
    );

    // A page that was not copied is not read as one: the control.
    rig.table(1, &[0, 1, 10, 9]);
    rig.restore(1, 0);
    let wrong = rig.step(1, 3 * PAGE, next, false);
    assert!(
        first_difference(&bytes(&want.logits), &bytes(&wrong.logits)).is_some(),
        "the third page matters to the rows after it"
    );
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
