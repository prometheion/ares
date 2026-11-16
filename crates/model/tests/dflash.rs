//! The DFlash drafter: its loader on the real files, and its pass on a device
//! against the CPU reference.
//!
//! The files: `ARES_TEST_DFLASH_27B` (Qwen3.8-27B-DFlash2), `ARES_TEST_DFLASH_FLASHNEXT`
//! (Qwen3.8-Flash-Next-DFlash2) and `ARES_TEST_DSPARK_27B` (the 27B's DSpark
//! drafter, which the DFlash kind refuses). A test skips the files that are not
//! named. The device test needs `ARES_TEST_CUDA=1` (`ARES_TEST_DEVICE` names the
//! device, default 0); its taps, embedding rows and head are random, so it
//! compares the drafter with the reference and nothing else.

use std::path::PathBuf;

use ares_core::Ledger;
use ares_cuda::{Device, DeviceBuffer, DevicePtr, Kernels, Stream, Wait};
use ares_gguf::Gguf;
use ares_model::dflash::{
    Cells, Config, Drafter, DrafterSetup, Error, PassSeq, Reference, Weights,
};

fn file(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os(name).map(PathBuf::from);
    if path.is_none() {
        eprintln!("skipped: set {name} to run this test on that file");
    }
    path
}

/// Normal numbers from a seed (xorshift and Box-Muller).
struct Normal(u64);

impl Normal {
    fn next(&mut self) -> f32 {
        let mut u = || {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (a, b) = (u(), u());
        ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

#[test]
fn the_drafter_files_load_and_other_variants_are_refused_by_name() {
    if let Some(path) = file("ARES_TEST_DFLASH_27B") {
        let gguf = Gguf::open(&path).unwrap();
        let c = Config::from_gguf(&gguf).unwrap();
        assert_eq!(
            (c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim),
            (5, 5120, 17408, 32, 8, 128)
        );
        assert_eq!(
            (c.block, c.top_k, c.selector_rank, c.conv_kernel, c.conv_group),
            (8, 16, 256, 2, 16)
        );
        assert_eq!(c.target_layers, [6, 20, 34, 48, 62]);
        assert_eq!((c.window, c.mask_token, c.n_vocab), (2048, 248070, 248320));
        assert_eq!(c.rope_base, 1e7);
        let w = Weights::load(&gguf, &c).unwrap();
        println!("27B drafter: {:.0} MiB of weights", w.bytes() as f64 / (1 << 20) as f64);
        c.fits(5120, 64, 248320).unwrap();
        assert!(matches!(c.fits(2560, 48, 248320), Err(Error::Target { .. })));
    }
    if let Some(path) = file("ARES_TEST_DFLASH_FLASHNEXT") {
        let gguf = Gguf::open(&path).unwrap();
        let c = Config::from_gguf(&gguf).unwrap();
        assert_eq!((c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_head_kv), (5, 2560, 7680, 32, 8));
        assert_eq!(c.target_layers, [6, 16, 26, 36, 46]);
        assert_eq!((c.block, c.mask_token, c.rope_base), (8, 248077, 1e4));
        let w = Weights::load(&gguf, &c).unwrap();
        println!("Flash-Next drafter: {:.0} MiB of weights", w.bytes() as f64 / (1 << 20) as f64);
        c.fits(2560, 48, 248320).unwrap();
    }
    if let Some(path) = file("ARES_TEST_DSPARK_27B") {
        let gguf = Gguf::open(&path).unwrap();
        match Config::from_gguf(&gguf) {
            Err(Error::Variant { what }) => {
                println!("DSpark refused: {what}");
                assert!(what.contains("DSpark"), "{what}");
            }
            other => panic!("the DSpark drafter was not refused as a variant: {other:?}"),
        }
    }
}

/// The share of `a`'s values that `b` has, over each position's candidates.
fn overlap(a: &[u32], b: &[u32]) -> usize {
    a.iter().filter(|t| b.contains(t)).count()
}

fn read_f32(buf: &DeviceBuffer, at: u64, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buf.read(at, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn upload(device: &Device, values: &[f32]) -> DeviceBuffer {
    let buf = DeviceBuffer::new(device, (values.len() * 4).max(16) as u64).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    buf.write(0, &bytes).unwrap();
    buf
}

/// One drafter on the device against the reference: a prompt of 50 rows, a
/// window of 8 rows of which 3 are accepted (the 5 rejected cells stay in the
/// ring and must not be read), and a pass after it.
fn device_against_reference(path: &PathBuf, device: &Device) {
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let (d, taps) = (c.n_embd, c.target_layers.len());
    let mut rng = Normal(0x5eed_d1a5);
    // The context: rows 0..50, then the window 50..58, of which 50..53 stay.
    const PROMPT: usize = 50;
    const WINDOW: usize = 8;
    const KEPT: usize = 3;
    let tap_rows: Vec<Vec<f32>> = (0..PROMPT + WINDOW).map(|_| rng.vec(taps * d, 1.0)).collect();
    let anchor = rng.vec(d, 0.02);
    let mask = rng.vec(d, 0.02);
    const HEAD: usize = 4096;
    let head = rng.vec(HEAD * d, 4.0 / (d as f32).sqrt());
    let head_of = |h: &[f32]| -> Vec<f32> {
        h.chunks_exact(d)
            .flat_map(|row| {
                head.chunks_exact(d)
                    .map(move |w| w.iter().zip(row).map(|(a, b)| a * b).sum::<f32>())
            })
            .collect()
    };
    let newest = 1234u32;
    let position = (PROMPT + KEPT) as u32;

    // The reference: the kept rows only.
    let reference = Reference::new(c.clone(), w.clone());
    let mut cells = Cells::default();
    let kept: Vec<f32> = tap_rows[..PROMPT + KEPT].concat();
    let positions: Vec<u32> = (0..(PROMPT + KEPT) as u32).collect();
    reference.inject(&mut cells, &kept, &positions);
    let lattice = reference.pass(&cells, 0, position, newest, (&anchor, &mask), &head_of, None);
    let chain = lattice.chain();

    // The device: every row, the rejected ones included.
    let kernels = Kernels::load(device).unwrap();
    let stream = Stream::new(device, 0).unwrap();
    let mut ledger = Ledger::new("test", device.mem_info().unwrap().free);
    let setup = DrafterSetup {
        max_slots: 2,
        max_rows: 64,
        window_rows: WINDOW as u32,
        head_rows: HEAD as u32,
    };
    let mut drafter = Drafter::open(&c, &w, device, &mut ledger, setup, mask.clone()).unwrap();
    for (from, rows) in [(0, PROMPT), (PROMPT, WINDOW)] {
        let step: Vec<(u32, u32)> = (from..from + rows).map(|p| (0, p as u32)).collect();
        drafter.begin_step(&stream, &step).unwrap();
        let mut held = Vec::new();
        for t in 0..taps {
            let values: Vec<f32> = (from..from + rows)
                .flat_map(|p| tap_rows[p][t * d..(t + 1) * d].to_vec())
                .collect();
            let buf = upload(device, &values);
            drafter.tap(&kernels, &stream, t, buf.ptr(), 1, 0, d as u32, 0, rows as u32).unwrap();
            held.push(buf);
        }
        drafter.inject(&kernels, &stream).unwrap();
        stream.synchronize().unwrap();
        drop(held);
    }
    let seqs = [PassSeq { slot: 0, position, newest, record_row: u32::MAX, wanted: 7, place: 0 }];
    let embed = |token: u32| {
        assert_eq!(token, newest);
        anchor.clone()
    };
    let (h_at, rows) = drafter.pass_layers(&kernels, &stream, &seqs, &embed).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(rows as usize, c.drafts());
    let (_, lattice_at, h_off, logits_at) = drafter.lattice();
    let scratch = drafter.lattice().0;
    assert_eq!(scratch.ptr().offset(h_off), h_at);
    let h = read_f32(scratch, h_off, rows as usize * d);
    // The final hidden states, against the reference's rows 1 .. block.
    let want = &lattice.hidden[d..];
    let err: f32 = h.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
    let norm: f32 = want.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("{}: hidden states off by {:.4} (rms, relative)", path.display(), err / norm);
    assert!(err / norm < 0.03, "the hidden states are off by {}", err / norm);

    // The candidates and the walk from the device's own hidden states.
    let logits = head_of(&h);
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    scratch.write(logits_at, &bytes).unwrap();
    let drafts = DeviceBuffer::new(device, 64).unwrap();
    drafter.pass_select(&kernels, &stream, DevicePtr::NULL, DevicePtr::NULL, drafts.ptr()).unwrap();
    stream.synchronize().unwrap();
    let k = c.top_k;
    let stride = k + k * k;
    let scratch = drafter.lattice().0;
    let got = read_f32(scratch, lattice_at, c.drafts() * stride);
    let mut drafted = vec![0u8; c.drafts() * 4];
    drafts.read(0, &mut drafted).unwrap();
    let drafted: Vec<u32> =
        drafted.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
    let mut same = 0;
    for j in 0..c.drafts() {
        let tokens: Vec<u32> = got[j * stride..j * stride + k].iter().map(|t| *t as u32).collect();
        let shared = overlap(&tokens, &lattice.tokens[j]);
        println!(
            "position {}: {shared} of {k} candidates shared; draft {} (p {:.3}), reference {} (p {:.3})",
            j + 1,
            drafted[j],
            drafter.prob(j as u32),
            chain[j].0,
            chain[j].1
        );
        assert!(shared + 3 >= k, "position {}: {tokens:?} against {:?}", j + 1, lattice.tokens[j]);
        same += usize::from(drafted[j] == chain[j].0);
    }
    assert!(same + 1 >= c.drafts(), "{same} of {} drafts are the reference's", c.drafts());
}

#[test]
fn the_drafter_on_the_device_drafts_what_the_reference_drafts() {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return;
    }
    let ordinal = std::env::var("ARES_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let device = Device::open(ordinal, Wait::Block).unwrap();
    for name in ["ARES_TEST_DFLASH_FLASHNEXT", "ARES_TEST_DFLASH_27B"] {
        if let Some(path) = file(name) {
            device_against_reference(&path, &device);
        }
    }
}
