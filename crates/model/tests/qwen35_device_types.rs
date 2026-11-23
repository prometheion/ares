//! The device implementation of `qwen35` on a made-up model whose weights, embedding table
//! included, are in the 2-bit types of the small file of Qwen3.8-27B (IQ2_XXS and IQ1_S, the
//! bulk of it, and Q2_K) or in one of the float types (F32, F16, BF16). Their prompts go
//! through the decode kernels up to the rows where the tile kernels are faster (for the float
//! types, up to one launch of eight rows) and through the tile kernels above, and the
//! embedding is gathered in the type it is stored in. A float model with weights that the tile
//! kernels cannot hold as f16 takes the kernels that can, matrix by matrix.
//!
//! Needs `ARES_TEST_CUDA=1` (and nothing else: the model is made here), and says so when it
//! skips. `ARES_TEST_DEVICE` picks the GPU (default 0). Lines that start with `ERR` are the
//! measurements.

mod common;

use ares_core::GgmlType;
use ares_cuda::{Device, Precision, Segment};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Harness, Options};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
use common::gpu::gpu;
use common::helpers::{Rng, rel_l2};
use common::{Spec, Synth, TempDir};

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
/// blocks of the 256-value types, with every matrix in `ty` but the two small ones of the
/// recurrent blocks, which are in `gates`: Q8_0, as in the real file, makes the matrices that
/// read one input be of two types. `large` puts weights into those two matrices of a float
/// model that an f16 tile cannot multiply (their outputs go through a sigmoid and a decay, so
/// the model stays a model): a row of `ssm_beta` of block 0 at 61440, which f16 holds but whose
/// sums of 64 products overflow f16, and two weights of 7e4 of opposite signs in a row of
/// `ssm_alpha` of block 1, which are infinities as f16.
fn model(ty: GgmlType, gates: GgmlType, large: bool) -> Vec<u8> {
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
    let mut edits = 0;
    for t in &synth.tensors {
        // The taps of the convolution are a matrix of f32 that no product reads.
        let matrix =
            t.name.ends_with(".weight") && t.dims.len() == 2 && !t.name.contains("ssm_conv1d");
        let gate = t.name.contains("ssm_alpha") || t.name.contains("ssm_beta");
        let ty = if gate { gates } else { ty };
        let mut values = t.values.clone();
        let n_in = t.dims[0] as usize;
        if large && t.name == "blk.0.ssm_beta.weight" {
            values[2 * n_in..3 * n_in].fill(61440.0);
            edits += 1;
        }
        if large && t.name == "blk.1.ssm_alpha.weight" {
            values[4 * n_in + 100] = -7.0e4;
            values[4 * n_in + 101] = 7.0e4;
            edits += 1;
        }
        builder = if !matrix || ty == GgmlType::F32 {
            // The values as they are: nothing is encoded.
            builder.tensor_f32(&t.name, &t.dims, &values)
        } else if matches!(ty, GgmlType::F16 | GgmlType::BF16 | GgmlType::Q8_0) {
            let mut bytes = vec![0u8; ty.bytes_for(values.len() as u64).unwrap() as usize];
            ares_quant::quantize(ty, &values, &mut bytes).unwrap();
            builder.tensor(&t.name, &t.dims, ty, bytes)
        } else {
            let n_blocks = values.len() / ty.block_values() as usize;
            builder.tensor(&t.name, &t.dims, ty, random_blocks(ty, n_blocks, &mut rng))
        };
    }
    assert_eq!(edits, if large { 2 } else { 0 }, "the model has the matrices the test edits");
    builder.build()
}

/// What the logits of a float model may differ by from the reference, three times what was
/// measured (RTX 4070, 2026-11-12). With every matrix in the float type: 2.4e-3 up to eight
/// rows, where the decode kernels multiply the weights and the activations as they are and the
/// rest of the device path is what differs, and 5.3e-3 to 7.1e-3 above, where the tile kernels
/// round both to f16 and sum in f16. A decoder that misplaces a value is of order 1, one that
/// loses the low bits of the weights of order 0.1.
const FLOAT_BOUND: f64 = 0.02;
/// The same with the two small matrices of the recurrent blocks in Q8_0, whose kernels
/// quantize the activations to int8: 1.3e-2 measured.
const MIXED_BOUND: f64 = 0.04;

fn direct(n: usize, pos0: usize) -> [Segment; 1] {
    [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: pos0 as i32, direct: true }]
}

/// The logits of prompts of `rows` rows each, and of one more row after each, on the device
/// against the reference: the worst row's relative distance.
fn worst_logits(device: &Device, model: &[u8], what: &str, rows: &[usize]) -> f64 {
    let dir = TempDir::new("types");
    let gguf = Gguf::open(dir.file("model.gguf", model)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let ids: Vec<u32> = (0..41).map(|i| ((i * 37 + 11) % 256) as u32).collect();
    let mut worst_of_all = 0.0f64;
    for &n in rows {
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
            let e = rel_l2(&logits[row * vocab..][..vocab], want);
            // A maximum passes over what is not a number.
            assert!(e.is_finite(), "{what} n={n} row {row}: logits {e}");
            worst = worst.max(e);
        }
        eprintln!("ERR logits {what} n={n}: worst row {worst:.2e} from the reference");
        worst_of_all = worst_of_all.max(worst);

        // One more row after them, from the state they left.
        if n > 1 {
            let next = ids[n];
            let window = reference.forward(&mut state, &[next], LogitRows::Last).unwrap();
            let logits = harness.step(&direct(1, n), &[next], &[0]).unwrap();
            let e = rel_l2(&logits, window.logits_of(0).unwrap());
            eprintln!("ERR logits {what} n=1 after {n} rows: {e:.2e} from the reference");
            assert!(e.is_finite(), "{what} after {n} rows: logits {e}");
            worst_of_all = worst_of_all.max(e);
        }
    }
    worst_of_all
}

/// Rows of one step: a decode step, the most one launch takes, the least that takes two, an
/// odd number and more than a step's worth of tensor-core rows.
const ROWS: [usize; 5] = [1, 8, 9, 21, 40];

#[test]
fn a_model_in_two_bit_and_float_types_runs_prompts_of_any_length() {
    let Some(device) = gpu() else { return };
    for ty in [GgmlType::IQ2_XXS, GgmlType::IQ1_S, GgmlType::Q2_K] {
        let worst = worst_logits(device, &model(ty, GgmlType::Q8_0, false), ty.name(), &ROWS);
        // The decode kernels of the quantized types quantize the activations to int8 and the
        // logits of a made-up model are flat, so a few percent differ; a wrong decoder is of
        // order 1.
        assert!(worst < 0.25, "{ty}: logits {worst}");
    }
    for ty in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
        // With the two small matrices in Q8_0, whose kernels quantize the activations to
        // int8: that is what the logits differ by.
        let what = format!("{ty} with q8_0");
        let worst = worst_logits(device, &model(ty, GgmlType::Q8_0, false), &what, &ROWS);
        assert!(worst < MIXED_BOUND, "{what}: logits {worst}");
        // Every matrix in the float type: up to eight rows the decode kernels multiply the
        // weights and the activations as they are, above that the tile kernels round both to
        // f16.
        let worst = worst_logits(device, &model(ty, ty, false), ty.name(), &ROWS);
        assert!(worst < FLOAT_BOUND, "{ty}: logits {worst}");
    }
}

/// Weights beyond what an f16 tile multiplies do not make the result worse: the matrix whose
/// sums of 64 products would overflow f16 is multiplied with f32 sums, and the group with the
/// matrix that f16 cannot hold by the decode kernels in launches. The logits are those of the
/// reference as closely as without such weights. (With the tile kernels as they are chosen
/// for other float matrices, the logits of the nine rows are not numbers.)
#[test]
fn float_weights_beyond_f16_are_multiplied_by_the_kernels_that_hold_them() {
    let Some(device) = gpu() else { return };
    for ty in [GgmlType::F32, GgmlType::BF16] {
        let what = format!("{ty} with large weights");
        let worst = worst_logits(device, &model(ty, ty, true), &what, &[8, 9, 40]);
        assert!(worst < FLOAT_BOUND, "{what}: logits {worst}");
    }
}
