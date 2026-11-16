//! What the prefix cache needs of the model's MTP block, on the real
//! Qwen3.8-27B's block: only the block, the embedding rows it keeps and the
//! output head go to the device (two gigabytes), and the hidden states the model
//! would compute are made up by the test.
//!
//! The block keeps state per sequence: its keys and values in the same pages
//! as the model's attention blocks (one more array of entries), and the final
//! hidden state of the sequence's last row (its carry), which the keys and
//! values of the next row and the next chain of drafts start from. So:
//!
//! - **a checkpoint holds the carry**: a sequence that restores it, reads the
//!   copy of the partly used page and goes on gets the keys and values, the
//!   chain of drafts and the outputs of the links the first sequence got,
//!   bit for bit; one that restores nothing does not;
//! - **a replay writes none of the block's cells** and still leaves the carry
//!   the next row starts from.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_27B`; `ARES_TEST_DEVICE`
//! names the device (default 0).

use std::ops::Range;
use std::path::PathBuf;

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, ENTRY_CELLS, Kernels, Precision, Segment, StepRecord, Stream, Wait,
};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{
    MtpChain, MtpStep, Options, PAGE_ENTRY, Part, PartSetup, Placement, StepEnv, page_bytes,
};
use ares_model::qwen35::{Config, Weights};
use ares_runtime::PinnedRam;
use ares_sampler::SampleParams;

const PAGE: u32 = ENTRY_CELLS;

/// The MTP block on a device, with the buffers the runtime would own: two
/// slots and a pool of sixteen pages.
struct Rig {
    stream: Stream,
    part: Part,
    record: DeviceBuffer,
    tables: DeviceBuffer,
    pages: DeviceBuffer,
    table_stride: u32,
    d: usize,
}

/// What a chain of drafts left: the tokens and the output of every link.
#[derive(Debug, PartialEq)]
struct Chain {
    drafts: Vec<u32>,
    outputs: Vec<Vec<u8>>,
}

impl Rig {
    fn open(device: &Device, c: &Config, w: &Weights<'_>, placement: &Placement) -> Self {
        let options = Options {
            precision: Precision::Exact,
            max_rows: 256,
            logit_rows: 8,
            window_rows: 8,
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
            d: c.n_embd,
        }
    }

    fn table(&mut self, slot: u32, pages: &[u32]) {
        let entries: Vec<u32> = pages.iter().map(|p| p | PAGE_ENTRY).collect();
        let bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
        self.tables.write(u64::from(slot * self.table_stride) * 4, &bytes).unwrap();
        self.part.update_table(slot, 0, entries.iter().copied()).unwrap();
    }

    /// Uploads a record of one segment and returns it with the offset of its
    /// token array.
    fn upload(&self, segment: Segment, tokens: &[u32]) -> (StepRecord, u64) {
        let record = StepRecord::new(&[segment], tokens, None, &[]).unwrap();
        let bytes = record.as_bytes();
        self.record.write(0, bytes).unwrap();
        let off_tokens = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        (record, u64::from(off_tokens))
    }

    /// Follows `tokens` from `pos0` in `slot` with `hidden` as the model's final
    /// hidden states of those rows; the rows are a replay if `replay`.
    fn follow(&mut self, slot: u32, pos0: u32, tokens: &[u32], hidden: &[f32], replay: bool) {
        let n = tokens.len() as u32;
        let segment = Segment { row0: 0, n_rows: n, slot, pos0: pos0 as i32, direct: true };
        let (record, _) = self.upload(segment, tokens);
        let segments = [segment];
        let rows: Vec<Range<u32>> =
            if replay { std::iter::once(0..n).collect() } else { Vec::new() };
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
        self.part.enqueue_mtp(&env).unwrap();
        self.part.enqueue_follow(&env, hidden).unwrap();
        self.stream.synchronize().unwrap();
    }

    /// Runs a chain of `links` drafts after `token` at `pos0` in `slot`.
    fn chain(&mut self, slot: u32, pos0: u32, token: u32, links: usize) -> Chain {
        let segment =
            Segment { row0: 0, n_rows: links as u32 + 1, slot, pos0: pos0 as i32, direct: false };
        let mut tokens = vec![0u32; links + 1];
        tokens[0] = token;
        let (record, off_tokens) = self.upload(segment, &tokens);
        let segments = [segment];
        let env = StepEnv {
            stream: &self.stream,
            step: self.record.ptr(),
            dims: record.dims(),
            tokens: &tokens,
            segments: &segments,
            tables: self.tables.ptr(),
            pages: self.pages.ptr(),
            exchange: None,
            replay: &[],
        };
        let rows = vec![SampleParams::GREEDY.row(0); links];
        self.part
            .mtp_step(MtpStep {
                tokens: self.record.ptr().offset(off_tokens),
                chains: vec![MtpChain { segment: 0, token, rows, needs: Vec::new(), place: 0 }],
                chances: None,
            })
            .unwrap();
        self.part.enqueue_mtp(&env).unwrap();
        self.stream.synchronize().unwrap();
        let drafted = self.part.drafted().expect("the step drafted");
        let (scratch, _) = self.part.mtp_scratch(0).unwrap();
        let mut bytes = vec![0u8; links * 4];
        scratch.read(drafted.ptr.as_raw() - scratch.ptr().as_raw(), &mut bytes).unwrap();
        let drafts =
            bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect::<Vec<_>>();
        let outputs = (0..links as u32)
            .map(|link| {
                let (scratch, [out, ..]) = self.part.mtp_scratch(link).unwrap();
                let mut bytes = vec![0u8; self.d * 4];
                scratch.read(out, &mut bytes).unwrap();
                bytes
            })
            .collect();
        Chain { drafts, outputs }
    }

    /// The carry row of `slot`.
    fn carry(&self, slot: u32) -> Vec<u8> {
        let (carry, rows) = self.part.mtp_carry().expect("the part has the block");
        let mut bytes = vec![0u8; self.d * 4];
        carry.read(u64::from(slot) * u64::from(rows) * self.d as u64 * 4, &mut bytes).unwrap();
        bytes
    }

    /// The block's cells: the converted entries of the pages, and the exact
    /// entries its tiles are in while they are exact.
    fn pool(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.pages.len() as usize];
        self.pages.read(0, &mut bytes).unwrap();
        if let Some(exact) = self.part.mtp_exact_entries() {
            let mut more = vec![0u8; exact.len() as usize];
            exact.read(0, &mut more).unwrap();
            bytes.extend(more);
        }
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

/// Numbers that look like hidden states and tokens, from a seed.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn hidden(&mut self, rows: usize, d: usize) -> Vec<f32> {
        (0..rows * d).map(|_| (self.next() % 2001) as f32 / 1000.0 - 1.0).collect()
    }

    fn tokens(&mut self, rows: usize) -> Vec<u32> {
        (0..rows).map(|_| (self.next() % 1000) as u32 + 10).collect()
    }
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Setup {
    device: Device,
    config: Config,
    gguf: Gguf,
}

fn setup() -> Option<Setup> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    let Some(path) = std::env::var_os("ARES_TEST_QWEN35_27B").map(PathBuf::from) else {
        eprintln!("skipped: set ARES_TEST_QWEN35_27B to run this test");
        return None;
    };
    let ordinal = std::env::var("ARES_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let gguf = Gguf::open(&path).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    Some(Setup { device: Device::open(ordinal, Wait::Block).unwrap(), config, gguf })
}

/// Only the MTP block, the embedding rows it keeps and the head on the device.
fn placement(c: &Config) -> Placement {
    let mut placement = Placement::single(c);
    placement.devices[0].blocks = vec![None; c.n_layer];
    placement.with_mtp(c, None).unwrap()
}

#[test]
fn a_sequence_that_restores_the_mtp_blocks_carry_and_reads_a_copied_page_drafts_as_the_first_did() {
    let Some(s) = setup() else { return };
    let weights = Weights::load(&s.gguf, &s.config).unwrap();
    let placement = placement(&s.config);
    let mut rig = Rig::open(&s.device, &s.config, &weights, &placement);
    let d = rig.d;
    assert_eq!(rig.part.checkpoint_bytes(), d as u64 * 4, "a checkpoint of the block is its carry");
    let mut noise = Noise(0x2545_f491_4f6c_dd1d);
    let (prefix, next) = (300usize, 60usize);
    let tokens = noise.tokens(prefix + next);
    let hidden = noise.hidden(prefix + next, d);
    let (head, tail) = (&tokens[..prefix], &tokens[prefix..]);
    let (hidden_head, hidden_tail) = (&hidden[..prefix * d], &hidden[prefix * d..]);

    // Slot 0 follows 300 rows (two whole pages and 44 cells of a third), a
    // checkpoint is taken, and it goes on with 60 rows and drafts.
    rig.table(0, &[0, 1, 2, 3]);
    for (i, rows) in head.chunks(150).enumerate() {
        let at = i * 150;
        rig.follow(0, at as u32, rows, &hidden_head[at * d..(at + rows.len()) * d], false);
    }
    assert_eq!(rig.carry(0), bytes(&hidden_head[(prefix - 1) * d..]), "the carry is the last row");
    rig.save(0, 1);
    rig.follow(0, prefix as u32, tail, hidden_tail, false);
    let want = rig.chain(0, (prefix + next) as u32, 77, 3);

    // Slot 1 reads the first two pages as they are and the third from a copy,
    // restores the checkpoint and goes on with the same 60 rows.
    rig.copy_page(2, 6);
    rig.table(1, &[0, 1, 6, 7]);
    rig.restore(1, 1);
    assert_eq!(rig.carry(1), bytes(&hidden_head[(prefix - 1) * d..]), "the carry was restored");
    rig.follow(1, prefix as u32, tail, hidden_tail, false);
    let got = rig.chain(1, (prefix + next) as u32, 77, 3);
    assert_eq!(got.drafts, want.drafts);
    for (link, (a, b)) in want.outputs.iter().zip(&got.outputs).enumerate() {
        let at = first_difference(a, b);
        assert_eq!(at, None, "link {link} parts at byte {at:?}");
    }
    eprintln!(
        "RES the MTP block drafted {:?} after a restored carry and a copied page, as it did \
         without, with the outputs of {} links bit for bit",
        got.drafts,
        got.outputs.len()
    );

    // The controls: without the restore the first new cell is computed from
    // another hidden state, and without the copy the cells of the third page
    // are missing. Both change what the chain computes.
    let other = noise.hidden(8, d);
    rig.table(1, &[8, 9, 10, 11]);
    rig.follow(1, 0, &noise.tokens(8), &other, false);
    rig.table(1, &[0, 1, 6, 7]);
    rig.follow(1, prefix as u32, tail, hidden_tail, false);
    let no_restore = rig.chain(1, (prefix + next) as u32, 77, 3);
    assert_ne!(no_restore.outputs, want.outputs, "the carry matters to the cell it feeds");
    rig.restore(1, 1);
    rig.table(1, &[0, 1, 12, 7]);
    rig.follow(1, prefix as u32, tail, hidden_tail, false);
    let no_copy = rig.chain(1, (prefix + next) as u32, 77, 3);
    assert_ne!(no_copy.outputs, want.outputs, "the cells of the copied page matter");
}

#[test]
fn a_replay_leaves_the_mtp_blocks_cells_alone_and_its_carry_at_the_last_row() {
    let Some(s) = setup() else { return };
    let weights = Weights::load(&s.gguf, &s.config).unwrap();
    let placement = placement(&s.config);
    let mut rig = Rig::open(&s.device, &s.config, &weights, &placement);
    let d = rig.d;
    let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
    let rows = PAGE as usize;
    let (tokens, hidden) = (noise.tokens(rows), noise.hidden(rows, d));

    rig.table(0, &[0, 1, 2, 3]);
    rig.follow(0, 0, &tokens, &hidden, false);
    let pool = rig.pool();
    assert!(pool.iter().any(|b| *b != 0), "the rows wrote cells");

    // Other rows over the same cells as a replay: nothing is written, and the
    // carry is the last of their hidden states.
    let (other_tokens, other_hidden) = (noise.tokens(rows), noise.hidden(rows, d));
    rig.follow(0, 0, &other_tokens, &other_hidden, true);
    let at = first_difference(&pool, &rig.pool());
    assert_eq!(at, None, "a replay wrote a cell of the block at byte {at:?}");
    assert_eq!(rig.carry(0), bytes(&other_hidden[(rows - 1) * d..]));

    // The control: the same rows written change the cells.
    rig.follow(0, 0, &other_tokens, &other_hidden, false);
    assert!(first_difference(&pool, &rig.pool()).is_some());
    eprintln!(
        "RES a replay of {rows} rows left the block's cells of {} bytes as they were",
        pool.len()
    );
}
