//! The runtime on real devices: the same scenarios as `tests/conductor.rs`,
//! with kernels computing the picks, for one device, a tensor split and a
//! layer split.
//!
//! Opening a device takes VRAM that a running model may need, so these run
//! only with `ARES_TEST_CUDA=1`, one at a time:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-runtime --test gpu -- --test-threads=1
//! ```
//!
//! `ARES_LIBCUDA=<path>/libcuda.so.1` selects the patched driver library, with
//! which the two-device placements use peer access and without which they go
//! through host memory.
//!
//! What they prove, beyond the conductor's logic: the next token of a
//! sequence is the same whatever else runs in the step, bit for bit, on each
//! placement (the placements differ from each other, which is expected, as
//! their sums add in a different order), and the threads, streams and
//! exchanges between devices neither deadlock nor lose a step.

mod common;

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ares_api::{BiasKey, FinishReason, LogitBias, LogprobRequest, SamplerKind, Sampling};
use ares_cuda::{Device, Wait, device_count};
use ares_runtime::{
    GpuConfig, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel, TokenFilter,
};
use ares_sampler::TokenMask;
use common::*;

fn gpus() -> Option<&'static [Device]> {
    static DEVICES: OnceLock<Option<Vec<Device>>> = OnceLock::new();
    DEVICES
        .get_or_init(|| {
            if std::env::var_os("ARES_TEST_CUDA").is_none() {
                eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
                return None;
            }
            let count = device_count().expect("the CUDA driver");
            assert!(count > 0, "ARES_TEST_CUDA is set but the driver reports no device");
            Some((0..count).map(|i| Device::open(i, Wait::Yield).expect("open device")).collect())
        })
        .as_deref()
}

/// The placements to test with the devices there are.
fn placements() -> Vec<(Split, usize)> {
    let count = gpus().map_or(0, <[Device]>::len);
    let mut all = vec![(Split::Single, 1)];
    if count >= 2 {
        all.push((Split::Tensor, 2));
        all.push((Split::Layer, 2));
    }
    all
}

struct Setup {
    model: SyntheticConfig,
    runtime: RuntimeConfig,
}

fn setup(split: Split, devices: usize) -> Setup {
    let mut model = SyntheticConfig::small(split, devices);
    model.context_length = 1024;
    Setup {
        model,
        runtime: RuntimeConfig {
            pages: 4096,
            cells_per_page: 16,
            max_slots: 64,
            ..RuntimeConfig::default()
        },
    }
}

fn start(setup: Setup) -> Runtime {
    let devices = gpus().expect("enabled");
    let used = setup.model.devices;
    let model = Arc::new(SyntheticModel::new(setup.model).unwrap());
    let gpu = GpuConfig::new(&setup.runtime, &*model);
    Runtime::start_gpu(model, devices[..used].to_vec(), setup.runtime, gpu)
        .expect("the runtime starts")
}

fn on_every_placement(check: impl Fn(Split, usize)) {
    if gpus().is_none() {
        return;
    }
    for (split, devices) in placements() {
        eprintln!("-- {split:?} over {devices} device(s)");
        check(split, devices);
    }
}

#[test]
fn output_is_independent_of_the_batch_on_every_placement() {
    on_every_placement(|split, devices| {
        for stagger in [Duration::ZERO, Duration::from_micros(300)] {
            let rt = start(setup(split, devices));
            output_is_independent_of_the_batch(&rt, stagger);
        }
    });
}

#[test]
fn the_text_depends_on_the_whole_history_on_every_placement() {
    on_every_placement(|split, devices| {
        let rt = start(setup(split, devices));
        // The same last token at the same position, a different first one.
        let a = alone(&rt, "aXz", Some(40)).text;
        let b = alone(&rt, "bXz", Some(40)).text;
        assert_ne!(a, b, "the history reaches the kernels");
        let distinct: std::collections::HashSet<char> = a.chars().collect();
        assert!(distinct.len() > 10, "not a degenerate output: {a}");
        // And the same twice.
        assert_eq!(alone(&rt, "aXz", Some(40)).text, a);
    });
}

#[test]
fn cancelling_frees_the_pages_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 4096;
        cancel_frees_pages_and_spares_the_others(&start(s));
    });
}

#[test]
fn more_requests_than_the_pool_holds_wait_in_line_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 256;
        // A request of up to 40 prompt tokens and 100 of output is charged
        // 3 pages and 7: three of them fit in 32 pages.
        s.runtime.pages = 32;
        more_requests_than_fit_wait_and_say_the_same(&start(s), 3);
    });
}

#[test]
fn a_request_that_gives_way_comes_back_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 256;
        s.runtime.pages = 24;
        s.runtime.output_ratio = 0.05;
        a_request_that_gives_way_comes_back_and_says_the_same(&start(s));
    });
}

#[test]
fn shutdown_with_requests_in_flight_joins_every_thread_on_every_placement() {
    on_every_placement(|split, devices| {
        let mut s = setup(split, devices);
        s.model.context_length = 4096;
        s.runtime.pages = 64 * 256;
        shutdown_ends_everything(&start(s));
        // A thread that has been joined is listed for a few microseconds more,
        // until the kernel has reaped it.
        let begun = std::time::Instant::now();
        while !engine_threads().is_empty() && begun.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_micros(200));
        }
        assert!(engine_threads().is_empty(), "threads left: {:?}", engine_threads());
    });
}

#[test]
fn requests_end_for_the_right_reason_on_every_placement() {
    on_every_placement(|split, devices| {
        let rt = start(setup(split, devices));
        requests_end_for_the_right_reason(&rt, 1024);
        bad_requests_are_refused_at_once(&rt, 1024);
    });
}

#[test]
fn the_end_token_ends_a_request_where_it_first_comes_up_on_every_placement() {
    on_every_placement(|split, devices| {
        the_end_token_ends_a_request_where_it_first_comes_up(&|eos| {
            let mut s = setup(split, devices);
            s.model.eos = eos;
            start(s)
        });
    });
}

#[test]
fn a_model_that_does_not_fit_is_refused_at_load_with_the_account() {
    let Some(devices) = gpus() else { return };
    let mut config = SyntheticConfig::small(Split::Single, 1);
    // Weights of about 40 GB.
    config.layers = 1;
    config.distinct_layers = 1;
    config.width = 51_200;
    config.attn_inner = 51_200;
    config.ffn_inner = 409_600;
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let runtime = RuntimeConfig { max_slots: 64, ..RuntimeConfig::default() };
    let gpu = GpuConfig::new(&runtime, &*model);
    let error = Runtime::start_gpu(model, devices[..1].to_vec(), runtime, gpu).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("synthetic weights") && text.contains("CUDA0"), "{text}");
}

#[test]
fn state_pages_are_reserved_before_the_model_is_built_and_do_not_change_the_text() {
    let Some(devices) = gpus() else { return };
    // A model that declares a kilobyte of state per cell: the pool's pages exist
    // on the device and the text is what it is without them.
    let plain = {
        let rt = start(setup(Split::Single, 1));
        outputs_of(&rt, 8, 3)
    };
    let mut s = setup(Split::Single, 1);
    s.model.state_bytes_per_cell = 1024;
    let rt = start(s);
    assert_eq!(outputs_of(&rt, 8, 3), plain);
    drop(rt);

    // A kilobyte more than the device holds is refused at load, and the
    // account names the pages.
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.state_bytes_per_cell = 1 << 30;
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let runtime = RuntimeConfig { max_slots: 64, ..RuntimeConfig::default() };
    let gpu = GpuConfig::new(&runtime, &*model);
    let error = Runtime::start_gpu(model, devices[..1].to_vec(), runtime, gpu).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("state pages") && text.contains("CUDA0"), "{text}");
}

/// Allows a window of 40 byte tokens that slides with the number of tokens
/// the reply and the drafts so far hold.
struct Sliding;

impl TokenFilter for Sliding {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        let start = 40 + ((reply.len() + drafts.len()) % 25) as u32;
        mask.fill(false);
        for token in start..start + 40 {
            mask.set(token, true);
        }
        true
    }
}

/// The settings a request can ask for, one kind of feature each: whether the
/// request is of a kind that verifies drafts, its settings, and whether it
/// has a filter.
fn features() -> Vec<(&'static str, bool, Sampling)> {
    let order = |kinds: &[SamplerKind]| Some(kinds.to_vec());
    let sampled = |f: &dyn Fn(&mut Sampling)| {
        let mut s = Sampling { temperature: Some(0.8), seed: Some(77), ..Sampling::default() };
        f(&mut s);
        s
    };
    vec![
        (
            "penalties",
            true,
            sampled(&|s| {
                s.top_k = Some(40);
                s.repeat_penalty = Some(1.3);
                s.frequency_penalty = Some(0.4);
                s.presence_penalty = Some(0.3);
                s.repeat_last_n = Some(48);
            }),
        ),
        (
            "dry",
            true,
            sampled(&|s| {
                s.dry_multiplier = Some(0.8);
                s.dry_allowed_length = Some(2);
                s.dry_penalty_last_n = Some(64);
                s.dry_sequence_breakers = Some(vec![" ".into()]);
            }),
        ),
        (
            "a bias and a ban",
            true,
            sampled(&|s| {
                s.logit_bias = vec![
                    LogitBias { key: BiasKey::Token(65), bias: Some(4.0) },
                    LogitBias { key: BiasKey::Token(66), bias: None },
                ];
            }),
        ),
        (
            "typical, top-n-sigma, xtc and a dynamic temperature",
            true,
            sampled(&|s| {
                s.typical_p = Some(0.9);
                s.top_n_sigma = Some(2.5);
                s.xtc_probability = Some(0.5);
                s.xtc_threshold = Some(0.2);
                s.dynatemp_range = Some(0.4);
                s.dynatemp_exponent = Some(1.5);
            }),
        ),
        (
            "an order of its own",
            true,
            sampled(&|s| {
                s.top_p = Some(0.9);
                s.min_p = Some(0.02);
                s.top_k = Some(100);
                s.samplers = order(&[
                    SamplerKind::Temperature,
                    SamplerKind::MinP,
                    SamplerKind::TopP,
                    SamplerKind::TopK,
                ]);
            }),
        ),
        (
            "the model's own probabilities",
            true,
            sampled(&|s| {
                s.top_p = Some(0.9);
                s.logprobs = Some(LogprobRequest { top: 5, after_chain: false });
            }),
        ),
        (
            "the probabilities after the chain",
            true,
            sampled(&|s| {
                s.top_p = Some(0.9);
                s.min_p = Some(0.05);
                s.logprobs = Some(LogprobRequest { top: 5, after_chain: true });
            }),
        ),
        (
            "greedy with penalties and probabilities",
            true,
            Sampling {
                temperature: Some(0.0),
                repeat_penalty: Some(1.4),
                repeat_last_n: Some(16),
                logprobs: Some(LogprobRequest { top: 3, after_chain: false }),
                ..Sampling::default()
            },
        ),
        ("mirostat 1", false, sampled(&|s| s.mirostat = Some(1))),
        (
            "mirostat 2 with probabilities",
            false,
            sampled(&|s| {
                s.mirostat = Some(2);
                s.mirostat_tau = Some(3.0);
                s.logprobs = Some(LogprobRequest { top: 4, after_chain: true });
            }),
        ),
        (
            "everything at once",
            true,
            sampled(&|s| {
                s.top_k = Some(60);
                s.top_p = Some(0.95);
                s.typical_p = Some(0.95);
                s.repeat_penalty = Some(1.2);
                s.frequency_penalty = Some(0.2);
                s.repeat_last_n = Some(32);
                s.dry_multiplier = Some(0.5);
                s.logit_bias = vec![LogitBias { key: BiasKey::Text("ab".into()), bias: Some(1.5) }];
                s.logprobs = Some(LogprobRequest { top: 2, after_chain: false });
            }),
        ),
    ]
}

fn feature_request(prompt: &str, sampling: Sampling) -> ares_api::GenerateRequest {
    let mut r = request(prompt, Some(150));
    r.sampling = sampling;
    r
}

/// A synthetic model with text that repeats and a draft block of its own, which
/// is what lets a window verify drafts at all (the n-gram drafter
/// takes part next to a model drafter).
fn drafting(split: Split, devices: usize) -> Setup {
    let mut s = setup(split, devices);
    s.model = s.model.repetitive(1, 40).with_own_drafts(4, &[]);
    s
}

/// A request that has every feature runs the same with drafts and without
/// (`draft: false`): the tokens, the text, the probabilities, on every
/// placement.
#[test]
fn every_sampler_feature_gives_the_same_tokens_with_and_without_drafts_on_every_placement() {
    on_every_placement(|split, devices| {
        let runtime = start(drafting(split, devices));
        let prompt = "the same words, the same words, the same words again";
        let mut accepted = 0;
        for (name, drafts, sampling) in features() {
            let mut plain = feature_request(prompt, sampling.clone());
            plain.draft = false;
            let before = snapshot(&runtime);
            let expected = run_one(&runtime, plain);
            assert_eq!(expected.usage.completion_tokens, 150, "{name}");
            assert_eq!(
                since(&snapshot(&runtime), &before)["draft_rows"],
                0,
                "{name}: draft: false"
            );
            if sampling.logprobs.is_some() {
                assert!(
                    expected.probs.len() >= 140,
                    "{name}: {} probabilities",
                    expected.probs.len()
                );
            }
            let before = snapshot(&runtime);
            let got = run_one(&runtime, feature_request(prompt, sampling));
            assert_eq!(got, expected, "{name}");
            let d = since(&snapshot(&runtime), &before);
            if drafts {
                assert!(d["draft_rows"] > 0, "{name}: windows verified drafts ({d:?})");
                accepted += d["accepted"];
            } else {
                assert_eq!(d["draft_rows"], 0, "{name}: this kind takes no drafts");
            }
        }
        assert!(accepted > 30, "drafts were accepted under the features ({accepted})");
        // A rule that restricts the picks.
        let mut r = feature_request(
            prompt,
            Sampling { temperature: Some(0.9), seed: Some(5), ..Sampling::default() },
        );
        r.id = fresh_id();
        r.draft = false;
        let expected = start_filtered(&runtime, r.clone(), Sliding).unwrap().finish();
        assert!(
            expected.text.chars().all(|c| (c as u32) >= 40 && (c as u32) < 105),
            "{}",
            expected.text
        );
        let before = snapshot(&runtime);
        r.id = fresh_id();
        r.draft = true;
        assert_eq!(start_filtered(&runtime, r, Sliding).unwrap().finish(), expected);
        let d = since(&snapshot(&runtime), &before);
        assert!(d["accepted"] > 10, "drafts are accepted under a filter ({d:?})");
        runtime.shutdown();
        drop(runtime);

        // A grammar, which refuses drafts as well as picks: the mask of every
        // row of a window is the mask of its own path, and a draft the grammar
        // forbids is cut. The text is the grammar's, with drafts and without.
        //
        // The grammar is whole after 140 tokens, and then nothing but the end
        // of the reply may follow. The model above has no token that ends a
        // reply, and a request for more than 140 tokens fails on it by name
        // (`a_reply_the_grammar_cannot_end_ends_in_an_error_and_not_in_free_text`
        // in `tests/structured.rs`). This one has the token beyond the bytes,
        // which the last word of a mask holds alone: the mask after the 140
        // tokens allows that token only, and the reply ends there by itself.
        let mut ending = drafting(split, devices);
        ending.model.eos = Some(256);
        let runtime = start(ending);
        let grammar = ares_api::Structure::new(ares_api::Shape::Gbnf(
            "root ::= (\"ab\" | \"cd\"){20} [a-f]{100}".into(),
        ));
        let mut r = feature_request(
            prompt,
            Sampling { temperature: Some(0.9), seed: Some(5), ..Sampling::default() },
        );
        r.structure = Some(grammar);
        r.draft = false;
        r.id = fresh_id();
        let expected = run_one(&runtime, r.clone());
        assert!(
            expected.text.chars().take(40).all(|c| ('a'..='d').contains(&c)),
            "{}",
            expected.text
        );
        assert!(expected.text.chars().all(|c| ('a'..='f').contains(&c)), "{}", expected.text);
        assert_eq!(
            (expected.text.len(), expected.usage.completion_tokens, expected.finish),
            (140, 141, FinishReason::Stop),
            "the grammar's 140 tokens and the token that ends the reply: {}",
            expected.text
        );
        let before = snapshot(&runtime);
        r.id = fresh_id();
        r.draft = true;
        assert_eq!(run_one(&runtime, r), expected, "the text of a grammar, with drafts");
        let d = since(&snapshot(&runtime), &before);
        eprintln!(
            "RES a grammar on {split:?}: {} rows of drafts verified, {} accepted, in {} steps",
            d["draft_rows"], d["accepted"], d["steps"]
        );
        assert!(d["draft_rows"] > 0, "windows verified drafts under a grammar ({d:?})");
        assert!(d["accepted"] > 10, "drafts are accepted under a grammar ({d:?})");
    });
}

/// Requests with different settings in one step do not touch each other.
#[test]
fn requests_with_different_settings_in_one_step_do_not_affect_each_other_on_every_placement() {
    on_every_placement(|split, devices| {
        let rt = start(drafting(split, devices));
        let prompt = |i: usize| format!("prompt number {i}, with some words in it");
        let kinds = features();
        let alone: Vec<Run> = kinds
            .iter()
            .enumerate()
            .map(|(i, (_, _, s))| run_one(&rt, feature_request(&prompt(i), s.clone())))
            .collect();
        // All of them at once, and plain requests among them.
        let mut handles = Vec::new();
        for (i, (_, _, s)) in kinds.iter().enumerate() {
            handles.push(common::start(&rt, feature_request(&prompt(i), s.clone())).unwrap());
            handles.push(common::start(&rt, request(&prompt(100 + i), Some(60))).unwrap());
        }
        let together: Vec<Run> = handles.into_iter().map(Handle::finish).collect();
        for (i, (name, _, _)) in kinds.iter().enumerate() {
            assert_eq!(together[2 * i], alone[i], "{name}");
        }
    });
}

/// Binds the last three tokens of every `period` tokens of a reply to the
/// letters from x to z and leaves the others free.
struct Stretches {
    period: usize,
}

impl Stretches {
    fn binds(&self, place: usize) -> bool {
        place % self.period >= self.period - 3
    }
}

impl TokenFilter for Stretches {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        if !self.binds(reply.len() + drafts.len()) {
            return false;
        }
        mask.fill(false);
        for token in b'x'..=b'z' {
            mask.set(u32::from(token), true);
        }
        true
    }

    fn free_rows(&mut self, reply: &[u32], tokens: &[u32]) -> usize {
        (0..=tokens.len()).take_while(|k| !self.binds(reply.len() + k)).count()
    }
}

/// A request with a rule that is free for stretches of its reply takes the
/// model's own drafts in them: the windows carry no mask, and what a window
/// gives ends at the token before a bound one. The tokens are the same with
/// drafts and without, on every placement.
///
/// The text never repeats, so the n-gram drafter has nothing to
/// offer. The synthetic model's draft block drafts what it has seen the model
/// pick, so the reply is made once without drafts on one runtime, and again
/// with drafts on a second runtime of the same model: its block knows the
/// reply, bound tokens included, and its n-gram drafter knows nothing.
#[test]
fn a_rule_that_is_free_in_stretches_takes_the_models_drafts_on_every_placement() {
    on_every_placement(|split, devices| {
        for period in [7usize, 10, 13] {
            let mut s = setup(split, devices);
            s.model = s.model.repetitive(1, 1).with_own_drafts(4, &[]);
            let used = s.model.devices;
            let model = Arc::new(SyntheticModel::new(s.model).unwrap());
            let open = || {
                let gpu = GpuConfig::new(&s.runtime, &*model);
                let devices = gpus().expect("enabled")[..used].to_vec();
                Runtime::start_gpu(model.clone(), devices, s.runtime.clone(), gpu)
                    .expect("the runtime starts")
            };
            let prompt = "free and bound in turn";
            let ask = |rt: &Runtime, draft: bool| {
                let mut r = request(prompt, Some(150));
                r.draft = draft;
                start_filtered(rt, r, Stretches { period }).unwrap().finish()
            };
            let first = open();
            let expected = ask(&first, false);
            assert_eq!(expected.usage.completion_tokens, 150, "stretches of {period}");
            let letters = expected.text.chars().filter(|c| ('x'..='z').contains(c)).count();
            assert!(letters >= 3 * (150 / period), "stretches of {period}: {}", expected.text);
            first.shutdown();
            drop(first);

            let second = open();
            let got = ask(&second, true);
            assert_eq!(got, expected, "stretches of {period}: drafts changed the reply");
            let c = snapshot(&second).counters;
            let (windows, accepted): (u64, u64) =
                (c.draft_lengths[1].iter().sum(), c.drafts_accepted_at[1].iter().sum());
            eprintln!(
                "RES stretches of {period}: {windows} windows took the block's drafts, {accepted} \
                 of its drafts accepted, {} rows of drafts cut again",
                c.drafts_rejected
            );
            assert!(
                accepted > 10,
                "stretches of {period}: {accepted} drafts of the block accepted"
            );
        }
    });
}
