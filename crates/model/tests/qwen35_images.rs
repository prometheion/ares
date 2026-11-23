//! An image's rows in `qwen35` on one GPU against the CPU reference: rows that
//! are not token embeddings, copied over the gathered ones as the runtime's
//! image stage is, at the rotary positions llama.cpp's `mtmd` gives an image
//! and the text after it, on the real Qwen3.5-0.8B with made-up rows.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL` (the 0.8B file), and
//! says so when it skips. `ARES_TEST_DEVICE` picks the GPU (default 0).

use std::sync::Arc;

use ares_cuda::{CacheFormat, Precision, Segment};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Harness, Options};
use ares_model::qwen35::{Config, Inputs, LogitRows, Reference, SequenceState, Weights};
use ares_runtime::{ImageInput, ImageSpan, Spans};
use common::gpu::gpu;
use common::helpers::rel_l2;

mod common;

/// The prompt: text, an image of 3 x 2 rows between its markers, text; then
/// rows of text one at a time after it.
#[test]
fn an_image_s_rows_at_their_rotary_positions_match_the_reference() {
    let Some(device) = gpu() else { return };
    let Some(path) = std::env::var_os("ARES_TEST_QWEN35_MODEL") else {
        eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to run this test");
        return;
    };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);

    // Twelve rows: four of text, the start marker, six image rows, the end
    // marker; the image's rows are made of two token embeddings each, so their
    // size is a token's.
    let (nx, ny, first) = (3u32, 2u32, 5u32);
    let mut tokens: Vec<u32> = vec![1000, 2000, 3000, 4000, 248_053];
    tokens.extend(std::iter::repeat_n(248_056, (nx * ny) as usize));
    tokens.push(248_054);
    let n = tokens.len();
    let embedding = |t: u32| {
        let mut row = vec![0.0f32; c.n_embd];
        w.token_embd.decode_row(t as usize, &mut row);
        row
    };
    let image_rows: Vec<(u32, Vec<f32>)> = (0..nx * ny)
        .map(|j| {
            let (a, b) = (embedding(500 + 37 * j), embedding(7000 + 11 * j));
            (first + j, a.iter().zip(&b).map(|(x, y)| 0.7 * x + 0.9 * y).collect())
        })
        .collect();
    let spans = Spans::new(vec![ImageSpan {
        first,
        image: Arc::new(ImageInput {
            name: [1; 32],
            width: nx * 32,
            height: ny * 32,
            pixels: Vec::new(),
            nx,
            ny,
        }),
    }]);
    let rotary3: Vec<[usize; 3]> =
        (0..n as u32 + 8).map(|cell| spans.rotary(cell).map(|p| p as usize)).collect();
    let rotary4: Vec<[i32; 4]> =
        rotary3.iter().map(|[t, h, w]| [*t as i32, *h as i32, *w as i32, 0]).collect();
    assert_eq!(rotary3[first as usize + 4], [5, 6, 6], "row 4 of 3 x 2: (p, p + 1, p + 1)");
    assert_eq!(rotary3[n - 1], [8, 8, 8], "the end marker is at p + max(nx, ny)");

    // The reference: the prompt in one window, then eight rows of text.
    let inputs = Inputs {
        rows: image_rows.iter().map(|(r, v)| (*r as usize, v.clone())).collect(),
        rotary: Some(rotary3[..n].to_vec()),
    };
    let mut state = SequenceState::new(&c);
    let window = reference.window_with(&state, &tokens, &inputs, LogitRows::All).unwrap();
    reference.commit(&mut state, &window, n).unwrap();
    let mut ref_logits: Vec<Vec<f32>> =
        (0..n).map(|r| window.logits_of(r).unwrap().to_vec()).collect();
    let after: Vec<u32> = (0..8).map(|i| 900 + 13 * i).collect();
    for (i, token) in after.iter().enumerate() {
        let cell = n + i;
        let one = Inputs { rows: Vec::new(), rotary: Some(vec![rotary3[cell]]) };
        let window = reference.window_with(&state, &[*token], &one, LogitRows::All).unwrap();
        reference.commit(&mut state, &window, 1).unwrap();
        ref_logits.push(window.logits_of(0).unwrap().to_vec());
    }
    // Without the rotary positions the rows of the image and the text after it
    // are elsewhere: the largest change of a row's logits, which the test must
    // be able to tell from the device's error.
    let plain = Inputs { rows: inputs.rows.clone(), rotary: None };
    let window =
        reference.window_with(&SequenceState::new(&c), &tokens, &plain, LogitRows::All).unwrap();
    let moved =
        (0..n).map(|r| rel_l2(window.logits_of(r).unwrap(), &ref_logits[r])).fold(0.0f64, f64::max);

    let opts = Options {
        precision: Precision::Exact,
        kv: (CacheFormat::F16, CacheFormat::F16),
        max_rows: 64,
        logit_rows: 16,
        window_rows: 16,
        ..Options::default()
    };
    let mut harness = Harness::open(device, &c, &w, &opts, 1, 512).unwrap();
    let all: Vec<u32> = (0..n as u32).collect();
    let segment = [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: 0, direct: true }];
    let logits =
        harness.step_with(&segment, &tokens, Some(&rotary4[..n]), &image_rows, &all).unwrap();
    let vocab = logits.len() / n;
    let mut worst = 0.0f64;
    for r in 0..n {
        worst = worst.max(rel_l2(&logits[r * vocab..(r + 1) * vocab], &ref_logits[r]));
    }
    for (i, token) in after.iter().enumerate() {
        let cell = n + i;
        let one = [Segment { row0: 0, n_rows: 1, slot: 0, pos0: cell as i32, direct: true }];
        let logits =
            harness.step_with(&one, &[*token], Some(&rotary4[cell..cell + 1]), &[], &[0]).unwrap();
        worst = worst.max(rel_l2(&logits, &ref_logits[cell]));
    }
    eprintln!(
        "ERR image rows and rotary positions: logits {worst:.2e} from the reference; the \
         reference with the rows at their cells is up to {moved:.2e} away"
    );
    assert!(worst < 2e-2, "{worst}");
    assert!(moved > 3.0 * worst, "the positions should matter: {moved} against {worst}");
}
