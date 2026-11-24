//! The `qwen4exp` reference on Qwen3.8-Flash-Next. The test needs the model
//! file (63.5 GiB in two parts, never in the repository) and skips with a
//! message unless it is told where it is:
//!
//! ```sh
//! ARES_TEST_FLASH_NEXT=<...>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
//!   cargo test --release -p ares-model --test qwen4exp_real -- --nocapture
//! ```
//!
//! It runs 26 tokens and took 224 seconds on a machine that was busy with
//! other work (opening the file and building the weight table: 0.1 seconds).

use ares_gguf::Gguf;
use ares_model::qwen4exp::{
    Config, LogitRows, MtpState, MtpWeights, Reference, SequenceState, Weights,
};
use common::helpers::argmax;

mod common;

/// `<|im_start|>user\nErkläre in drei Sätzen, warum der Himmel blau
/// ist.<|im_end|>\n<|im_start|>assistant\n`
const PROMPT: [u32; 22] = [
    248045, 846, 198, 18871, 10248, 45125, 303, 55593, 326, 179212, 11, 183857, 2607, 199908,
    242880, 5810, 13, 248046, 198, 248045, 74455, 198,
];

/// What the model continues with under greedy picks: `<think>\nThe user ...`.
/// Strata 0.1.48 gives the same tokens for this prompt (and the same 64
/// tokens after it; `docs/reference/qwen4exp-architecture.md`, section 10).
const CONTINUATION: [u32; 4] = [248068, 198, 760, 1156];

#[test]
fn flash_next_continues_a_german_prompt_as_strata_does() {
    let Some(path) = std::env::var_os("ARES_TEST_FLASH_NEXT") else {
        eprintln!("skipped: set ARES_TEST_FLASH_NEXT to the first part of the model file");
        return;
    };
    let started = std::time::Instant::now();
    let gguf = Gguf::open(&path).expect("the model file opens");
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    // The table is built without reading the experts or the n-gram table.
    eprintln!("configuration and weight table in {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!((config.n_layer, config.n_embd, config.n_stream), (48, 2560, 4));
    assert_eq!((config.experts.count, config.experts.used, config.experts.ff), (512, 10, 640));
    assert_eq!(config.n_attention_layers(), 12);
    assert_eq!(config.indexer.dense_limit(), 2051);
    let ngram = config.ngram.as_ref().expect("the model has the n-gram layer");
    assert_eq!((ngram.layer, ngram.n_heads(), ngram.row_dim), (1, 16, 160));
    assert_eq!(ngram.table_rows, 320_001_536);
    assert_eq!(weights.layers[0].moe.down.expert_bytes(), 460_800);

    let reference = Reference::new(&config, &weights).keep_residual(false);
    let mut state = SequenceState::new(&config);
    let window = reference.forward(&mut state, &PROMPT, LogitRows::Last).expect("prompt");
    let logits = window.logits_of(PROMPT.len() - 1).expect("logits of the last row");
    let mut next = argmax(logits);
    // 28.833 when this was written; Strata's value for the row is 29.597.
    let top = logits[next as usize];
    assert!((top - 28.833).abs() < 0.02, "the largest logit is {top}");
    for (i, &expected) in CONTINUATION.iter().enumerate() {
        assert_eq!(next, expected, "generated token {i}");
        let window = reference.forward(&mut state, &[next], LogitRows::Last).expect("token");
        next = argmax(window.logits_of(0).expect("logits"));
    }
    eprintln!("{} tokens in {:.0} s", state.position(), started.elapsed().as_secs_f64());
}

/// `<|im_start|>user\nExplain in three sentences how a hash table
/// works.<|im_end|>\n<|im_start|>assistant\n` and the 80 tokens the reference
/// continues with under greedy picks (an empty thinking block and three
/// sentences); Strata 0.1.48 writes the same 80.
const ENGLISH: [u32; 99] = [
    248045, 846, 198, 814, 20139, 303, 2250, 22157, 1204, 264, 5010, 1898, 4138, 13, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271, 32, 5010, 1898, 10220, 795, 539, 12366, 6679,
    310, 3050, 14505, 303, 449, 1287, 1608, 264, 5010, 709, 13, 3095, 264, 1328, 369, 20918, 11,
    279, 5010, 709, 55288, 1141, 1857, 11, 10377, 279, 1785, 310, 1925, 279, 869, 5774, 506, 421,
    3602, 364, 4778, 2528, 13, 1368, 1330, 6679, 5010, 310, 279, 1788, 1857, 11, 3750, 430, 264,
    18188, 11, 279, 1898, 5533, 12178, 1040, 71159, 466, 1724, 26153, 310, 3436, 279, 4945, 2663,
    13,
];

/// The tokens of [`ENGLISH`] that were given; the others the model wrote.
const ENGLISH_GIVEN: usize = 19;

/// What the MTP block of the real model rests on. The file of the block has
/// no outputs to compare with, so the test measures what the block is for: fed
/// the trunk's residual of a cell and the token behind it, how often its
/// largest logit is the token the trunk writes one further, on text the trunk
/// wrote itself, alone and in chains of three links. Strata's own probe of the
/// block reports 0.89, 0.86 and 0.85 for the three links (its `mtp.hpp`).
///
/// ```sh
/// ARES_TEST_FLASH_NEXT=<first part of the model> \
/// ARES_TEST_FLASH_NEXT_MTP=<...>/mtp-q2_0.gguf \
///   cargo test --release -p ares-model --test qwen4exp_real draft -- --nocapture
/// ```
///
/// `ARES_TEST_QWEN4EXP_DRAFT_TOKENS` runs fewer than the 99 tokens (a token of
/// the trunk takes about nine seconds on a machine that is busy).
#[test]
fn the_mtp_block_guesses_what_the_trunk_writes() {
    let (Some(path), Some(draft)) =
        (std::env::var_os("ARES_TEST_FLASH_NEXT"), std::env::var_os("ARES_TEST_FLASH_NEXT_MTP"))
    else {
        eprintln!(
            "skipped: set ARES_TEST_FLASH_NEXT to the first part of the model file and \
             ARES_TEST_FLASH_NEXT_MTP to the file of its MTP block"
        );
        return;
    };
    let n = std::env::var("ARES_TEST_QWEN4EXP_DRAFT_TOKENS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(ENGLISH.len())
        .clamp(ENGLISH_GIVEN + 8, ENGLISH.len());
    let started = std::time::Instant::now();
    let gguf = Gguf::open(&path).expect("the model file opens");
    let file = Gguf::open(&draft).expect("the file of the MTP block opens");
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let head = MtpWeights::load(&file, &config).expect("the MTP block");
    assert_eq!(head.layer.moe.gate.expert_bytes(), 460_800);
    assert_eq!(head.layer.moe.up.span(), (460_800, 921_600));

    let reference = Reference::new(&config, &weights).keep_residual(false);
    let tokens = &ENGLISH[..n];
    let wide = config.wide();
    let trunk = reference
        .window(&SequenceState::new(&config), tokens, LogitRows::None)
        .expect("the trunk")
        .trunk_out;
    eprintln!("the trunk, {n} tokens: {:.0} s", started.elapsed().as_secs_f64());

    // Every cell of the block that has its token: cell i pairs the residual
    // of cell i with token i + 1 and guesses token i + 2.
    let cells = n - 1;
    let follow = reference
        .draft_window(&head, &MtpState::new(), &trunk[..cells * wide], &tokens[1..], LogitRows::All)
        .expect("the block follows");
    let (mut right, mut asked) = (0, 0);
    for cell in ENGLISH_GIVEN - 2..cells - 1 {
        asked += 1;
        right += usize::from(follow.draft(cell) == Some(tokens[cell + 2]));
    }
    eprintln!(
        "one link, every cell that guesses a token the model wrote: {right} of {asked} \
         ({:.0} s)",
        started.elapsed().as_secs_f64()
    );

    // Chains of three links from every eighth cell: a link counts when the
    // links before it were right, as a verify window counts them.
    let (mut offered, mut accepted) = ([0usize; 3], [0usize; 3]);
    for cell in (ENGLISH_GIVEN - 2..cells.saturating_sub(3)).step_by(8) {
        let mut state = MtpState::new();
        reference.mtp_commit(&mut state, &follow, cell).expect("the cells before");
        let drafts = reference
            .draft_chain(&head, &state, &trunk[cell * wide..(cell + 1) * wide], tokens[cell + 1], 3)
            .expect("a chain");
        assert_eq!(Some(drafts[0]), follow.draft(cell), "link 0 is the cell the block follows");
        for (link, draft) in drafts.iter().enumerate() {
            offered[link] += 1;
            if *draft != tokens[cell + 2 + link] {
                break;
            }
            accepted[link] += 1;
        }
    }
    eprintln!(
        "chains of three links: accepted {accepted:?} of {offered:?} ({:.0} s)",
        started.elapsed().as_secs_f64()
    );
    // A block that is wired wrongly guesses next to nothing.
    assert!(right * 10 >= asked * 7, "{right} of {asked} single links are right");
    assert!(accepted[1] * 2 >= offered[1], "second links: {accepted:?} of {offered:?}");
}
