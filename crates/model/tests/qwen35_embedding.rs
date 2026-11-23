//! Where the token embedding of `qwen35` lies, on made-up models: in host memory, with the
//! device keeping only the rows its draft chain can pick, and a tied head (a file without
//! `output.weight`, whose head is the table) keeping its own.
//!
//! The placement tests need nothing; the device tests need `ARES_TEST_CUDA=1` (the models
//! are made here), `ARES_TEST_DEVICE` picks the GPU (default 0). Lines that start with `ERR`
//! are the measurements.

mod common;

use ares_core::GgmlType;
use ares_cuda::{Precision, Segment};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{EmbeddingReport, Harness, Options, Placement};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
use common::gpu::gpu;
use common::helpers::rel_l2;
use common::{Spec, Synth, TempDir};

/// What the MTP block may propose in a case.
#[derive(Clone, Debug)]
enum Drafts {
    /// The model has no MTP block placed.
    No,
    /// It proposes the tokens of a list.
    Listed(Vec<u32>),
    /// It proposes every token of the device's head.
    Any,
}

const MAX_ROWS: u32 = 32;

/// How far the logits of a prompt may be from the reference's, as the relative L2 distance
/// over all rows and over the worst row. The gather copies and decodes rows and rounds
/// nothing, so what remains is the rounding of the products, f16 operands on the tensor
/// cores against f32 on the host. Measured on an RTX 4070 (2026-11-12): 2.4e-3 to 2.7e-3 over
/// all rows and 3.9e-3 to 5.1e-3 in the worst row, for both types and the same wherever the
/// rows came from. One wrong row is 0.26 and 1.4 away (the test measures it), so the bounds
/// are four times the rounding and at most a twenty-fifth of a wrong row.
const ROUNDING: f64 = 1e-2;
const ROUNDING_OF_A_ROW: f64 = 2e-2;
const N_VOCAB: usize = 256;

/// A model of four blocks in a shape the device kernels run: the delta rule's heads are 128
/// wide, which is the one width its kernels are built for.
fn spec(tied: bool, nextn: bool, matrix_type: GgmlType) -> Spec {
    Spec {
        n_layer: 4,
        n_embd: 256,
        n_ff: 256,
        n_vocab: N_VOCAB,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 256,
        k_heads: 2,
        v_heads: 6,
        state: 128,
        nextn,
        tied,
        matrix_type,
        ..Spec::default()
    }
}

fn placement(c: &Config, drafts: &Drafts) -> Placement {
    let single = Placement::single(c);
    match drafts {
        Drafts::No => single,
        Drafts::Listed(tokens) => single.with_mtp(c, Some(tokens)).unwrap(),
        Drafts::Any => single.with_mtp(c, None).unwrap(),
    }
}

/// Every third token, in a different order than the table's.
fn listed(n_vocab: usize) -> Vec<u32> {
    let mut tokens: Vec<u32> = (0..n_vocab as u32).filter(|t| t % 3 == 1).collect();
    tokens.reverse();
    tokens
}

#[test]
fn the_rows_a_draft_chain_picks_are_the_draft_heads_and_the_host_stages_the_rest() {
    let dir = TempDir::new("embedding-placement");
    let synth = Synth::new(spec(false, true, GgmlType::F32));
    let gguf = Gguf::open(dir.file("model.gguf", &synth.builder().build())).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    // No MTP block: nothing is kept, every row of every step comes from the host.
    let none = placement(&c, &Drafts::No);
    assert_eq!((none.kept_rows(), none.stages_rows(&c)), (0, true));
    // A draft vocabulary: the rows of its tokens, no matter in which order they were listed.
    let list = listed(c.n_vocab);
    let some = placement(&c, &Drafts::Listed(list.clone()));
    assert_eq!((some.kept_rows(), some.stages_rows(&c)), (list.len() as u32, true));
    // Without one the chain can pick every token of the head on the device: the whole table
    // is kept, and the host stages nothing.
    let any = placement(&c, &Drafts::Any);
    assert_eq!((any.kept_rows(), any.stages_rows(&c)), (c.n_vocab as u32, false));
}

/// The logits of the reference for `ids`, one row of the vocabulary each.
fn reference_logits(c: &Config, w: &Weights<'_>, ids: &[u32]) -> Vec<f32> {
    let mut state = SequenceState::new(c);
    let window = Reference::new(c, w).forward(&mut state, ids, LogitRows::All).unwrap();
    (0..ids.len()).flat_map(|row| window.logits_of(row).unwrap().to_vec()).collect()
}

/// What the embedding of a part should be in a case.
fn expected(
    tied: bool,
    drafts: &Drafts,
    n_vocab: usize,
    table_bytes: u64,
    stage_bytes: u64,
) -> EmbeddingReport {
    let row = table_bytes / n_vocab as u64;
    let kept_rows = match drafts {
        Drafts::No => 0,
        Drafts::Listed(tokens) => tokens.len(),
        Drafts::Any => n_vocab,
    };
    let stages = kept_rows < n_vocab;
    EmbeddingReport {
        embeds: true,
        table_bytes,
        host_bytes: if stages { table_bytes + stage_bytes } else { 0 },
        kept_rows: kept_rows as u32,
        // A tied head holds the rows anyway: they are its first rows.
        kept_bytes: if tied { 0 } else { kept_rows as u64 * row },
        shared_with_head: tied && kept_rows > 0,
        staged_bytes: if stages { stage_bytes } else { 0 },
    }
}

#[test]
fn the_model_reads_its_prompt_from_host_rows_and_kept_rows_with_a_head_of_its_own_or_a_tied_one() {
    let Some(device) = gpu() else { return };
    let ids: Vec<u32> = (0..29).map(|i| ((i * 37 + 11) % N_VOCAB) as u32).collect();
    // The prompt has tokens of the list and tokens that are not.
    let list = listed(N_VOCAB);
    assert!(ids.iter().any(|t| list.contains(t)) && ids.iter().any(|t| !list.contains(t)));
    for matrix_type in [GgmlType::F32, GgmlType::Q8_0] {
        for tied in [false, true] {
            let synth = Synth::new(spec(tied, true, matrix_type));
            let dir = TempDir::new("embedding");
            let gguf = Gguf::open(dir.file("model.gguf", &synth.builder().build())).unwrap();
            let c = Config::from_gguf(&gguf).unwrap();
            let w = Weights::load(&gguf, &c).unwrap();
            assert_eq!(w.output.is_none(), tied);
            let want = reference_logits(&c, &w, &ids);
            // What one wrong row does to the logits, where it does least: another last
            // token, which changes the last row only. The bounds lie far below it.
            let mut other = ids.clone();
            *other.last_mut().unwrap() = (ids[ids.len() - 1] + 1) % N_VOCAB as u32;
            let wrong = reference_logits(&c, &w, &other);
            let last = (ids.len() - 1) * c.n_vocab..;
            let apart = rel_l2(&wrong, &want);
            let apart_row = rel_l2(&wrong[last.clone()], &want[last]);
            eprintln!(
                "ERR one wrong row {matrix_type} tied {tied}: {apart:.2e}, its row {apart_row:.2e}"
            );
            assert!(
                apart > 10.0 * ROUNDING && apart_row > 10.0 * ROUNDING_OF_A_ROW,
                "{matrix_type} tied {tied}: one wrong row is {apart} and {apart_row} away"
            );
            for drafts in [Drafts::No, Drafts::Listed(list.clone()), Drafts::Any] {
                let placement = placement(&c, &drafts);
                let opts = Options {
                    precision: Precision::Exact,
                    max_rows: MAX_ROWS,
                    logit_rows: 32,
                    window_rows: if matches!(drafts, Drafts::No) { 0 } else { 8 },
                    ..Options::default()
                };
                let mut harness =
                    Harness::open_placed(device, &c, &w, &placement, &opts, 1, 512).unwrap();
                let what = format!("{matrix_type} tied {tied} {drafts:?}");

                let table_bytes = w.token_embd.bytes().len() as u64;
                let row_bytes = table_bytes / c.n_vocab as u64;
                let report = harness.part().embedding_report();
                let stage_bytes = u64::from(MAX_ROWS) * row_bytes;
                assert_eq!(
                    report,
                    expected(tied, &drafts, c.n_vocab, table_bytes, stage_bytes),
                    "{what}"
                );

                let n = ids.len();
                let out: Vec<u32> = (0..n as u32).collect();
                let segments =
                    [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: 0, direct: true }];
                let got = harness.step(&segments, &ids, &out).unwrap();
                let width = got.len() / n;
                // The columns of the logits are the head's rows: in the order of the draft
                // vocabulary when there is one.
                let mut ordered = vec![0.0f32; got.len()];
                for row in 0..n {
                    for column in 0..width {
                        let token = placement.head_token(column as u32) as usize;
                        ordered[row * c.n_vocab + token] = got[row * width + column];
                    }
                }
                assert_eq!(width, c.n_vocab, "{what}");
                let error = rel_l2(&ordered, &want);
                let worst = (0..n)
                    .map(|row| {
                        let at = row * c.n_vocab..(row + 1) * c.n_vocab;
                        rel_l2(&ordered[at.clone()], &want[at])
                    })
                    .fold(0.0, f64::max);
                eprintln!(
                    "ERR logits {what}: {error:.2e} from the reference, worst row {worst:.2e}"
                );
                assert!(error < ROUNDING && worst < ROUNDING_OF_A_ROW, "{what}: {error} {worst}");
            }
        }
    }
}
