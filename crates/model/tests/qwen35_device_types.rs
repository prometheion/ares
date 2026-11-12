//! The device implementation of `qwen35` on a made-up model whose weights, embedding table
//! included, are in the 2-bit types of the small file of Qwen3.8-27B: IQ2_XXS and IQ1_S
//! (the bulk of it), whose prompts go through the tile kernels, and Q2_K, which has none yet,
//! so that its prompts go through the decode kernels in launches of eight rows whatever the
//! number of rows. The embedding is gathered in the type it is stored in.
//!
//! Needs `ARES_TEST_CUDA=1` (and nothing else: the model is made here), and says so when it
//! skips. `ARES_TEST_DEVICE` picks the GPU (default 0). Lines that start with `ERR` are the
//! measurements.

mod common;

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_cuda::{Device, Precision, Segment, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Harness, Options};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `n_blocks` blocks of `ty` of random bytes with small, finite f16 scales: any other byte
/// is valid in these types.
fn random_blocks(ty: GgmlType, n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let scales: &[usize] = match ty {
        GgmlType::IQ2_XXS | GgmlType::IQ1_S => &[0],
        GgmlType::Q2_K => &[80, 82],
        _ => panic!("the test makes no blocks of {ty}"),
    };
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for b in bytes.chunks_exact_mut(block) {
        for &at in scales {
            let scale = 0.004 + 0.02 * ((rng.next() >> 40) as f32 / (1u64 << 24) as f32);
            b[at..at + 2].copy_from_slice(&ares_quant::f32_to_f16(scale).to_le_bytes());
        }
    }
    bytes
}

/// A model of four blocks whose widths, and the widths of the slices of a device, are whole
/// blocks of the 256-value types, with every matrix but the two small ones of the recurrent
/// blocks in `ty`; those are Q8_0, as they are in the real file, so that the matrices that
/// read one input are of two types, one of which has a tensor-core kernel.
fn model(ty: GgmlType) -> Vec<u8> {
    let spec = Spec {
        n_layer: 4,
        n_embd: 256,
        n_ff: 256,
        n_vocab: 256,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 256,
        k_heads: 2,
        v_heads: 6,
        state: 128,
        ..Spec::default()
    };
    let synth = Synth::new(spec);
    let mut rng = Rng(0x5EED_0007);
    let mut builder = synth.metadata(None);
    for t in &synth.tensors {
        // The taps of the convolution are a matrix of f32 that no product reads.
        let matrix =
            t.name.ends_with(".weight") && t.dims.len() == 2 && !t.name.contains("ssm_conv1d");
        builder = if !matrix {
            builder.tensor_f32(&t.name, &t.dims, &t.values)
        } else if t.name.contains("ssm_alpha") || t.name.contains("ssm_beta") {
            let mut bytes =
                vec![0u8; GgmlType::Q8_0.bytes_for(t.values.len() as u64).unwrap() as usize];
            ares_quant::quantize(GgmlType::Q8_0, &t.values, &mut bytes).unwrap();
            builder.tensor(&t.name, &t.dims, GgmlType::Q8_0, bytes)
        } else {
            let n_blocks = t.values.len() / ty.block_values() as usize;
            builder.tensor(&t.name, &t.dims, ty, random_blocks(ty, n_blocks, &mut rng))
        };
    }
    builder.build()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        d += (f64::from(*g) - f64::from(*w)).powi(2);
        n += f64::from(*w).powi(2);
    }
    (d / n).sqrt()
}

fn direct(n: usize, pos0: usize) -> [Segment; 1] {
    [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: pos0 as i32, direct: true }]
}

#[test]
fn a_model_in_two_bit_types_runs_prompts_of_any_length() {
    let Some(device) = gpu() else { return };
    for ty in [GgmlType::IQ2_XXS, GgmlType::IQ1_S, GgmlType::Q2_K] {
        let dir = TempDir::new("types");
        let gguf = Gguf::open(dir.file("model.gguf", &model(ty))).unwrap();
        let c = Config::from_gguf(&gguf).unwrap();
        let w = Weights::load(&gguf, &c).unwrap();
        let reference = Reference::new(&c, &w);
        let ids: Vec<u32> = (0..41).map(|i| ((i * 37 + 11) % 256) as u32).collect();
        // Rows of one step: a decode step, the most one launch takes, the least that takes
        // two, an odd number and more than a step's worth of tensor-core rows.
        for n in [1usize, 8, 9, 21, 40] {
            let mut state = SequenceState::new(&c);
            let window = reference.forward(&mut state, &ids[..n], LogitRows::All).unwrap();
            let opts = Options {
                precision: Precision::Fast,
                max_rows: 64,
                logit_rows: 40,
                window_rows: 16,
                ..Options::default()
            };
            let mut harness = Harness::open(device, &c, &w, &opts, 1, 512).unwrap();
            let out: Vec<u32> = (0..n as u32).collect();
            let logits = harness.step(&direct(n, 0), &ids[..n], &out).unwrap();
            let vocab = logits.len() / n;
            let mut worst = 0.0f64;
            for row in 0..n {
                let want = window.logits_of(row).unwrap();
                worst = worst.max(rel_l2(&logits[row * vocab..][..vocab], want));
            }
            eprintln!("ERR logits {ty} n={n}: worst row {worst:.2e} from the reference");
            // The decode kernels quantize the activations to int8 and the logits of a made-up
            // model are flat, so a few percent differ; a wrong decoder is of order 1.
            assert!(worst < 0.25, "{ty} n={n}: logits {worst}");

            // One more row after them, from the state they left.
            if n > 1 {
                let next = ids[n];
                let window = reference.forward(&mut state, &[next], LogitRows::Last).unwrap();
                let logits = harness.step(&direct(1, n), &[next], &[0]).unwrap();
                let e = rel_l2(&logits, window.logits_of(0).unwrap());
                eprintln!("ERR logits {ty} n=1 after {n} rows: {e:.2e} from the reference");
                assert!(e < 0.25, "{ty} after {n} rows: logits {e}");
            }
        }
    }
}
