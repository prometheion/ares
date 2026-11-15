//! A row of a draft window computes with the bits it has alone: `qwen35` on a
//! device, with the fast kernels, on made-up models.
//!
//! A reply must not depend on what was drafted for it. A step verifies a
//! window of rows of a sequence (its newest token and drafts), and the drafts
//! that equal the picks are accepted, so row `k` of a window has to give the
//! logits that a step of that row alone gives after the `k` rows before it
//! were committed, bit for bit. The tests run the same tokens through two
//! sequences, one a row at a time and one in windows of several lengths, some
//! with a wrong draft that is rejected, and compare every logit of every
//! accepted row, the recurrent state and the stored keys and values:
//!
//! - for models of recurrent blocks only, of attention blocks only and of
//!   both, so that a difference names its kind of block;
//! - for every format of K and V (`f16`, `q8`, and `q7` to `q4`, whose tiles
//!   are converted while the windows run), with float matrices and with
//!   `Q8_0` ones (8-bit activations);
//! - block by block, every block fed the rows the single rows fed it.
//!
//! Needs `ARES_TEST_CUDA=1` (the models are made here) and says so when it
//! skips. `ARES_TEST_DEVICE` picks the GPU (default 0).

mod common;

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_cuda::{CacheFormat, Device, Precision, Segment, StateFormat, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Harness, Options};
use ares_model::qwen35::{Config, Weights};
use common::{Spec, Synth, TempDir};

fn gpu() -> Option<&'static Device> {
    static DEVICE: OnceLock<Option<Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            if std::env::var_os("ARES_TEST_CUDA").is_none() {
                eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
                return None;
            }
            let ordinal =
                std::env::var("ARES_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            Some(Device::open(ordinal, Wait::Block).expect("open device"))
        })
        .as_ref()
}

/// Which blocks a made-up model has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Blocks {
    Recurrent,
    Attention,
    Both,
}

/// A model of four blocks at widths the device kernels take (heads of 256 for
/// attention, of 128 for the delta rule), two query heads per KV head.
fn model(blocks: Blocks, matrix_type: GgmlType) -> Vec<u8> {
    let spec = Spec {
        n_layer: 4,
        n_embd: 256,
        n_ff: 256,
        n_vocab: 256,
        n_ctx: 1024,
        n_head: 4,
        n_head_kv: 2,
        head_dim: 256,
        k_heads: 2,
        v_heads: 6,
        state: 128,
        interval: match blocks {
            Blocks::Recurrent => 5,
            Blocks::Attention => 1,
            Blocks::Both => 2,
        },
        matrix_type,
        ..Spec::default()
    };
    Synth::new(spec).builder().build()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 20) % n
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn segment(slot: u32, pos0: usize, rows: usize, direct: bool) -> [Segment; 1] {
    [Segment { row0: 0, n_rows: rows as u32, slot, pos0: pos0 as i32, direct }]
}

/// What a comparison covered.
#[derive(Clone, Copy, Debug, Default)]
struct Covered {
    rows: usize,
    windows: usize,
    rejected: usize,
}

/// Runs `tokens` through slot 0 a row at a time and through slot 1 in draft
/// windows of up to `longest` rows after a prompt of `prompt` rows, and
/// asserts that every accepted row of a window has the logits of the row
/// alone, and that the two sequences end with the same recurrent state and the
/// same stored keys and values.
#[allow(clippy::too_many_arguments)]
fn windows_equal_single_rows(
    device: &Device,
    file: &[u8],
    what: &str,
    opts: &Options,
    prompt: usize,
    tokens: &[u32],
    longest: usize,
    rng: &mut Rng,
) -> Covered {
    let dir = TempDir::new("windows");
    let gguf = Gguf::open(dir.file("model.gguf", file)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let cells = (tokens.len() + longest).next_multiple_of(128) as u32;
    let mut h = Harness::open(device, &c, &w, opts, 2, cells).unwrap();
    let vocab = c.n_vocab;
    for slot in 0..2 {
        for (i, chunk) in tokens[..prompt].chunks(64).enumerate() {
            h.step(&segment(slot, i * 64, chunk.len(), true), chunk, &[]).unwrap();
        }
    }
    // Slot 0: every row alone.
    let alone: Vec<Vec<f32>> = (prompt..tokens.len())
        .map(|p| h.step(&segment(0, p, 1, true), &tokens[p..p + 1], &[0]).unwrap())
        .collect();
    // Slot 1: windows. Some have a wrong draft: the rows from it on are
    // rejected and the next window starts at the first of them.
    let mut covered = Covered::default();
    let mut at = prompt;
    while at < tokens.len() {
        let rows = (1 + rng.below(longest as u64) as usize).min(tokens.len() - at);
        let mut window = tokens[at..at + rows].to_vec();
        let accepted = if rows > 1 && rng.below(3) == 0 {
            let wrong = 1 + rng.below(rows as u64 - 1) as usize;
            window[wrong] = (window[wrong] + 1) % vocab as u32;
            covered.rejected += 1;
            wrong
        } else {
            rows
        };
        let out: Vec<u32> = (0..rows as u32).collect();
        // A window of one row is a row without drafts: it needs no commit.
        let deferred = rows > 1;
        let logits = h.step(&segment(1, at, rows, !deferred), &window, &out).unwrap();
        if deferred {
            h.commit(&[accepted as u32]).unwrap();
        }
        for r in 0..accepted {
            assert_eq!(
                bits(&logits[r * vocab..(r + 1) * vocab]),
                bits(&alone[at + r - prompt]),
                "{what}: row {r} of a window of {rows} rows at position {at} is not the row alone"
            );
        }
        covered.rows += accepted;
        covered.windows += usize::from(deferred);
        at += accepted;
    }
    // One more row in both, so that both tables are of the same position, and
    // then what the sequences keep.
    let last = [tokens[0]];
    let a = h.step(&segment(0, tokens.len(), 1, true), &last, &[0]).unwrap();
    let b = h.step(&segment(1, tokens.len(), 1, true), &last, &[0]).unwrap();
    assert_eq!(bits(&a), bits(&b), "{what}: the row after the windows");
    for i in 0..h.part().n_blocks() {
        if h.part().kv_place(i, 0, 0).is_some() {
            let cells = tokens.len() as u32 + 1;
            assert!(
                h.read_kv(i, 0, cells).unwrap() == h.read_kv(i, 1, cells).unwrap(),
                "{what}: the stored K/V of block {i}"
            );
        } else {
            let (conv_a, delta_a) = h.read_recurrent_state(i, 0).unwrap();
            let (conv_b, delta_b) = h.read_recurrent_state(i, 1).unwrap();
            assert_eq!(bits(&conv_a), bits(&conv_b), "{what}: convolution windows of block {i}");
            assert_eq!(bits(&delta_a), bits(&delta_b), "{what}: delta matrices of block {i}");
        }
    }
    covered
}

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.below(256) as u32).collect()
}

const FORMATS: [CacheFormat; 6] = [
    CacheFormat::F16,
    CacheFormat::Q8,
    CacheFormat::Q7,
    CacheFormat::Q6,
    CacheFormat::Q5,
    CacheFormat::Q4,
];

fn options(kv: (CacheFormat, CacheFormat)) -> Options {
    Options {
        precision: Precision::Fast,
        kv,
        max_rows: 64,
        logit_rows: 16,
        window_rows: 16,
        ..Options::default()
    }
}

/// Every kind of block and every format of K and V: the rows of windows of up
/// to 8 rows against the rows alone, over 560 positions, so that the windows
/// cross the entries of the K/V tables and, with `q7` to `q4`, the positions
/// at which a tile is converted (with `q4`, whose 256 newest cells stay
/// exact, the first at position 512).
#[test]
fn a_row_of_a_window_has_the_logits_of_the_row_alone_for_every_block_and_format() {
    let Some(device) = gpu() else { return };
    let ids = tokens(560, 0x0051_0E57);
    let mut rng = Rng(0xD4AF_7001);
    for blocks in [Blocks::Recurrent, Blocks::Attention, Blocks::Both] {
        let file = model(blocks, GgmlType::F32);
        // A model without attention keeps no K and V: one format says it all.
        let formats: &[CacheFormat] =
            if blocks == Blocks::Recurrent { &FORMATS[1..2] } else { &FORMATS };
        for &format in formats {
            let what = format!("{blocks:?} blocks, K and V as {format:?}");
            let covered = windows_equal_single_rows(
                device,
                &file,
                &what,
                &options((format, format)),
                100,
                &ids,
                8,
                &mut rng,
            );
            eprintln!(
                "RES {what}: {} rows in {} windows ({} with a rejected draft) equal the rows alone",
                covered.rows, covered.windows, covered.rejected
            );
            assert!(covered.windows >= 60 && covered.rejected >= 15, "{what}: {covered:?}");
        }
    }
}

/// K and V in different formats, matrices of 8-bit weights (whose products
/// read 8-bit activations), the delta-rule matrices kept as `f16`, and windows
/// longer than one launch of the weight kernels takes.
#[test]
fn rows_of_windows_equal_the_rows_alone_with_mixed_formats_and_other_kernels() {
    let Some(device) = gpu() else { return };
    let ids = tokens(420, 0x000A_110E);
    let mut rng = Rng(0x00DD_5EED);
    let float = model(Blocks::Both, GgmlType::F32);
    let quantized = model(Blocks::Both, GgmlType::Q8_0);
    let pairs = [
        (CacheFormat::Q8, CacheFormat::Q4),
        (CacheFormat::Q4, CacheFormat::F16),
        (CacheFormat::F16, CacheFormat::Q8),
        (CacheFormat::Q6, CacheFormat::Q8),
    ];
    for kv in pairs {
        let what = format!("K as {:?}, V as {:?}", kv.0, kv.1);
        windows_equal_single_rows(device, &float, &what, &options(kv), 70, &ids, 8, &mut rng);
    }
    for kv in [(CacheFormat::Q8, CacheFormat::Q8), (CacheFormat::Q5, CacheFormat::Q5)] {
        let what = format!("Q8_0 matrices, K and V as {:?}", kv.0);
        windows_equal_single_rows(device, &quantized, &what, &options(kv), 70, &ids, 8, &mut rng);
    }
    let state = Options { state: StateFormat::F16, ..options((CacheFormat::Q8, CacheFormat::Q8)) };
    windows_equal_single_rows(device, &float, "f16 state", &state, 70, &ids, 8, &mut rng);
    windows_equal_single_rows(device, &quantized, "f16 state, Q8_0", &state, 70, &ids, 8, &mut rng);
    // Windows of up to 16 rows: more than one launch of the decode kernels.
    for (file, name) in [(&float, "float"), (&quantized, "Q8_0")] {
        let what = format!("windows of up to 16 rows, {name} matrices");
        let opts = options((CacheFormat::Q8, CacheFormat::Q8));
        windows_equal_single_rows(device, file, &what, &opts, 70, &ids, 16, &mut rng);
    }
}

/// Block by block: every block of a window fed the rows that the single rows
/// fed it gives their outputs, so a block that fails here is the one at
/// fault, whatever the blocks before it did.
#[test]
fn every_block_gives_a_row_of_a_window_what_it_gives_the_row_alone() {
    let Some(device) = gpu() else { return };
    let ids = tokens(300, 0xB10C);
    let file = model(Blocks::Both, GgmlType::F32);
    let dir = TempDir::new("blocks");
    let gguf = Gguf::open(dir.file("model.gguf", &file)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let d = c.n_embd;
    let embed = |tokens: &[u32]| -> Vec<f32> {
        let mut x = vec![0.0f32; tokens.len() * d];
        for (row, t) in x.chunks_mut(d).zip(tokens) {
            w.token_embd.decode_row(*t as usize, row);
        }
        x
    };
    for format in [CacheFormat::Q8, CacheFormat::Q4, CacheFormat::F16] {
        let mut h = Harness::open(device, &c, &w, &options((format, format)), 2, 512).unwrap();
        // 250 cells and a window of 8: it crosses the entry that begins at 256.
        let (prompt, rows) = (250usize, 8usize);
        for slot in 0..2 {
            for (i, chunk) in ids[..prompt].chunks(64).enumerate() {
                h.step(&segment(slot, i * 64, chunk.len(), true), chunk, &[]).unwrap();
            }
        }
        let window = &ids[prompt..prompt + rows];
        let alone: Vec<Vec<Vec<f32>>> = (0..rows)
            .map(|r| {
                let x = embed(&window[r..r + 1]);
                h.blocks(&segment(0, prompt + r, 1, true), |i| (i == 0).then(|| x.clone())).unwrap()
            })
            .collect();
        let together = h
            .blocks(&segment(1, prompt, rows, false), |i| {
                Some(if i == 0 {
                    embed(window)
                } else {
                    (0..rows).flat_map(|r| alone[r][i - 1].iter().copied()).collect()
                })
            })
            .unwrap();
        h.commit(&[rows as u32]).unwrap();
        for (i, block) in together.iter().enumerate() {
            let kind = if h.part().kv_place(i, 0, 0).is_some() { "attention" } else { "recurrent" };
            for r in 0..rows {
                assert_eq!(
                    bits(&block[r * d..(r + 1) * d]),
                    bits(&alone[r][i]),
                    "K and V as {format:?}: block {i} ({kind}), row {r} of the window"
                );
            }
        }
    }
}
