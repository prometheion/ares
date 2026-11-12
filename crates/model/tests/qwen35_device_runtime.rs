//! The device implementation of `qwen35` as the runtime drives it: the real
//! Qwen3.5-0.8B on one GPU, and split over two in both ways, through the
//! device threads, the step record and the links.
//!
//! Needs `ARES_TEST_CUDA=1`, `ARES_TEST_QWEN35_MODEL` and
//! `ARES_TEST_QWEN35_CAPTURE`; the split tests need two GPUs. Lines that start
//! with `RES` are the measurements.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use ares_api::{
    BiasKey, Engine, Event, EventSink, GenerateRequest, LogitBias, LogprobRequest, Message, Prompt,
    RequestId, Role, SamplerKind, Sampling, TokenProbs,
};
use ares_cuda::{Device, Precision, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Rows, Session, Spread, Wanted};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig, TokenFilter};
use ares_sampler::TokenMask;

#[path = "common/kv_env.rs"]
mod kv_env;

fn enabled() -> bool {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return false;
    }
    true
}

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set {name} to run this test");
            None
        }
    }
}

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

fn argmax(x: &[f32]) -> (u32, f32) {
    let mut best = 0;
    let mut second = f32::NEG_INFINITY;
    for i in 1..x.len() {
        if x[i] > x[best] {
            second = x[best];
            best = i;
        } else if x[i] > second {
            second = x[i];
        }
    }
    (best as u32, x[best] - second)
}

fn options(precision: Precision) -> Options {
    kv_env::with_env(Options {
        precision,
        max_rows: 64,
        logit_rows: 16,
        window_rows: 16,
        ..Options::default()
    })
}

/// The picks of every prompt row (steps of 16 rows) and 40 greedy tokens.
fn run(session: &mut Session, ids: &[u32]) -> (Vec<(u32, f32)>, Vec<u32>) {
    let mut rows = Vec::new();
    for (i, chunk) in ids.chunks(16).enumerate() {
        let part = Rows {
            slot: 0,
            position: (i * 16) as u32,
            tokens: chunk,
            direct: true,
            wanted: Wanted::All,
        };
        rows.extend(session.step(&[part]).unwrap().iter().map(|p| (p.token, p.logit)));
    }
    let mut generated = vec![rows[ids.len() - 1].0];
    while generated.len() < 40 {
        let token = *generated.last().unwrap();
        let part = Rows {
            slot: 0,
            position: (ids.len() + generated.len() - 1) as u32,
            tokens: &[token],
            direct: true,
            wanted: Wanted::Last,
        };
        generated.push(session.step(&[part]).unwrap()[0].token);
    }
    (rows, generated)
}

/// One device, a tensor split and a layer split of the same model against the
/// CPU reference: every placement picks what the reference picks, up to near
/// ties, and the two cards together write the text one card writes.
#[test]
fn every_placement_of_the_0_8b_agrees_with_the_reference() {
    let (true, Some(path), Some(capture)) =
        (enabled(), env_path("ARES_TEST_QWEN35_MODEL"), env_path("ARES_TEST_QWEN35_CAPTURE"))
    else {
        return;
    };
    let two = device_count().unwrap() >= 2;
    if !two {
        eprintln!("one GPU: the splits are not tested");
    }
    let gguf = Gguf::open(&path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    for language in ["german", "english"] {
        let ids = read_ids(&capture.join(language).join("base/prompt.tokens"));
        let n = ids.len();
        let mut state = SequenceState::new(&c);
        let window = reference.forward(&mut state, &ids, LogitRows::All).unwrap();
        let want: Vec<(u32, f32)> = (0..n).map(|r| argmax(window.logits_of(r).unwrap())).collect();
        for precision in [Precision::Exact, Precision::Fast] {
            let mut texts: Vec<(String, Vec<u32>)> = Vec::new();
            let mut placements = vec![("single", Spread::Single, vec![0u32])];
            if two {
                placements.push(("tensor", Spread::Tensor(vec![0.625, 0.375]), vec![0, 1]));
                placements.push(("layer", Spread::Layer(vec![0.5, 0.5]), vec![0, 1]));
            }
            for (name, spread, ordinals) in placements {
                let model = Arc::new(
                    Qwen35::new(Gguf::open(&path).unwrap(), "q", &spread, options(precision), 512)
                        .unwrap(),
                );
                let devices: Vec<Device> =
                    ordinals.iter().map(|d| Device::open(*d, Wait::Block).unwrap()).collect();
                let mut session =
                    Session::start(model.clone(), devices, 1, 512, 256 << 20).unwrap();
                let (rows, generated) = run(&mut session, &ids);
                let agree = rows.iter().zip(&want).filter(|(g, w)| g.0 == w.0).count();
                let widest_miss = rows
                    .iter()
                    .zip(&want)
                    .filter(|(g, w)| g.0 != w.0)
                    .map(|(_, w)| w.1)
                    .fold(0.0f32, f32::max);
                // The logit of the pick, where both pick the same token.
                let worst_logit = rows
                    .iter()
                    .zip(0..n)
                    .filter(|(g, r)| g.0 == want[*r].0)
                    .map(|(g, r)| {
                        let exact = window.logits_of(r).unwrap()[g.0 as usize];
                        ((g.1 - exact) / exact.abs().max(1.0)).abs()
                    })
                    .fold(0.0f32, f32::max);
                let weights: Vec<String> = model
                    .reports()
                    .iter()
                    .map(|r| format!("{:.0} MiB", r.weight_bytes as f64 / 1048576.0))
                    .collect();
                eprintln!(
                    "RES {language} {precision:?} {name}: top-1 {agree}/{n} with the reference (widest missed margin {widest_miss:.3}), worst logit of a pick {worst_logit:.2e} off, weights {}",
                    weights.join(" + ")
                );
                assert!(agree * 100 >= n * 90, "{name} {precision:?}: top-1 {agree}/{n}");
                texts.push((name.to_owned(), generated));
            }
            // The layer split again with the prompt in steps of 64 rows that
            // are handed over in groups of 16: the two cards then work on
            // different groups of a step at once.
            if two {
                let opts = Options { pipeline_rows: 16, ..options(precision) };
                let spread = Spread::Layer(vec![0.5, 0.5]);
                let model = Arc::new(
                    Qwen35::new(Gguf::open(&path).unwrap(), "q", &spread, opts, 512).unwrap(),
                );
                let devices: Vec<Device> =
                    (0..2).map(|d| Device::open(d, Wait::Block).unwrap()).collect();
                let mut session = Session::start(model, devices, 1, 512, 256 << 20).unwrap();
                let mut pick = 0;
                for (i, chunk) in ids.chunks(64).enumerate() {
                    let part = Rows {
                        slot: 0,
                        position: (i * 64) as u32,
                        tokens: chunk,
                        direct: true,
                        wanted: Wanted::Last,
                    };
                    pick = session.step(&[part]).unwrap()[0].token;
                }
                let mut generated = vec![pick];
                while generated.len() < 40 {
                    let token = *generated.last().unwrap();
                    let part = Rows {
                        slot: 0,
                        position: (n + generated.len() - 1) as u32,
                        tokens: &[token],
                        direct: true,
                        wanted: Wanted::Last,
                    };
                    generated.push(session.step(&[part]).unwrap()[0].token);
                }
                texts.push(("layer in groups".to_owned(), generated));
            }
            for (name, generated) in &texts[1..] {
                let same = generated.iter().zip(&texts[0].1).take_while(|(a, b)| a == b).count();
                eprintln!(
                    "RES {language} {precision:?} {name}: {same}/40 greedy tokens equal the single device's"
                );
            }
        }
    }
}

fn ask(runtime: &Runtime, id: u64, question: &str, max_tokens: u32) -> String {
    let (tx, rx) = mpsc::channel();
    let text = Arc::new(Mutex::new(String::new()));
    let sink = {
        let text = text.clone();
        EventSink::new(move |event| {
            match event {
                Event::Logprobs(_) => {}
                Event::Text(piece) => text.lock().unwrap().push_str(&piece),
                Event::Done { .. } => {
                    let _ = tx.send(Ok(()));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
            }
            true
        })
    };
    runtime
        .submit(
            GenerateRequest {
                model: "q".into(),
                prompt: Prompt::Chat {
                    messages: vec![Message::new(Role::User, question)],
                    tools: None,
                    variables: Vec::new(),
                },
                sampling: Sampling::default(),
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft: true,
                structure: None,
                id: RequestId(id),
            },
            sink,
        )
        .unwrap();
    rx.recv().unwrap().unwrap();
    std::mem::take(&mut *text.lock().unwrap())
}

/// The whole engine: chat requests through the conductor, alone and two at
/// once. (The 0.8B has no draft block, so nothing is drafted: the drafts of the
/// 27B are checked in `qwen35_mtp_runtime`.)
#[test]
fn the_engine_answers_alone_and_two_at_once() {
    let (true, Some(path)) = (enabled(), env_path("ARES_TEST_QWEN35_MODEL")) else {
        return;
    };
    let questions = [
        "Z\u{e4}hle von eins bis zwanzig, jede Zahl als Wort in einer eigenen Zeile, und dann noch einmal.",
        "Write the line 'the quick brown fox jumps over the lazy dog' four times, numbered.",
    ];
    let model = Arc::new(
        Qwen35::new(
            Gguf::open(&path).unwrap(),
            "q",
            &Spread::Single,
            options(Precision::Exact),
            1024,
        )
        .unwrap(),
    );
    let config =
        RuntimeConfig { pages: 32, cells_per_page: 128, max_slots: 2, ..RuntimeConfig::default() };
    let gpu = GpuConfig::new(&config, &*model);
    let device = Device::open(0, Wait::Block).unwrap();
    let runtime = Runtime::start_gpu(model, vec![device], config, gpu).unwrap();
    let mut texts = Vec::new();
    for (i, question) in questions.iter().enumerate() {
        let text = ask(&runtime, i as u64 + 1, question, 120);
        eprintln!("RES alone: {question}\n--- answer ---\n{text}\n---");
        assert!(text.chars().count() > 20, "an answer of a few words: {text:?}");
        texts.push(text);
    }
    // Both questions again, at once: two sequences share every step.
    let together: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = questions
            .iter()
            .enumerate()
            .map(|(i, question)| {
                let runtime = &runtime;
                scope.spawn(move || ask(runtime, 10 + i as u64, question, 120))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (alone, shared) in texts.iter().zip(&together) {
        let same = alone.chars().zip(shared.chars()).take_while(|(a, b)| a == b).count();
        eprintln!(
            "RES two sequences at once: {same} of {} characters equal the text of the sequence alone",
            alone.chars().count()
        );
        assert!(same >= 60, "a sequence next to another says something else: {shared:?}");
    }
    if let Some(snapshot) = runtime.snapshot() {
        eprintln!("RES counters: {:?}", snapshot.counters);
    }
    runtime.shutdown();
}

/// What a request said and the probabilities it reported.
type Answer = (String, Vec<TokenProbs>);

/// A request with the sampler settings `sampling`, restricted by `filter`
/// when there is one.
fn ask_with(
    runtime: &Runtime,
    id: u64,
    question: &str,
    sampling: Sampling,
    filter: Option<Box<dyn TokenFilter>>,
    max_tokens: u32,
) -> Answer {
    let (tx, rx) = mpsc::channel();
    let said = Arc::new(Mutex::new((String::new(), Vec::new())));
    let sink = {
        let said = said.clone();
        EventSink::new(move |event| {
            match event {
                Event::Logprobs(probs) => said.lock().unwrap().1.extend(probs),
                Event::Text(piece) => said.lock().unwrap().0.push_str(&piece),
                Event::Done { .. } => {
                    let _ = tx.send(Ok(()));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
            }
            true
        })
    };
    let request = GenerateRequest {
        model: "q".into(),
        prompt: Prompt::Chat {
            messages: vec![Message::new(Role::User, question)],
            tools: None,
            variables: Vec::new(),
        },
        sampling,
        max_tokens: Some(max_tokens),
        stop: Vec::new(),
        draft: true,
        structure: None,
        id: RequestId(id),
    };
    match filter {
        Some(filter) => runtime.submit_filtered(request, sink, filter).unwrap(),
        None => runtime.submit(request, sink).unwrap(),
    }
    rx.recv().unwrap().unwrap();
    std::mem::take(&mut *said.lock().unwrap())
}

/// Allows the tokens below 100 000 and, for every other row, only the even ones.
struct Evens(u32);

impl TokenFilter for Evens {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        let n = reply.len() + drafts.len();
        mask.fill(false);
        for token in (0..self.0).step_by(1 + n % 2) {
            mask.set(token, true);
        }
        true
    }
}

/// The probabilities of an answer that asked for `top` alternatives per token
/// (or for none), and the rule they follow.
fn check_probabilities(name: &str, probs: &[TokenProbs], top: Option<usize>) {
    let Some(top) = top else {
        assert!(probs.is_empty(), "{name}: probabilities nobody asked for");
        return;
    };
    assert!(probs.len() >= 20, "{name}: {} probabilities", probs.len());
    for token in probs {
        assert!(token.logprob <= 0.0 && token.logprob.is_finite(), "{name}: {token:?}");
        let n = token.alternatives.len();
        assert!((1..=top).contains(&n), "{name}: {n} alternatives of at most {top}: {token:?}");
        let best = token.alternatives.iter().map(|a| a.logprob).fold(f32::MIN, f32::max);
        assert!(token.logprob <= best + 1e-6, "{name}: no alternative is more probable: {token:?}");
    }
}

/// Every kind of sampler setting through the whole engine on the real model:
/// each answers, and the probabilities that were asked for come with it. (This
/// model has no draft block, so nothing is drafted; that drafts do not change
/// the answer under these settings is checked on the 27B, in
/// `qwen35_mtp_runtime`.)
#[test]
fn every_kind_of_sampler_setting_answers_on_the_real_model() {
    let (true, Some(path)) = (enabled(), env_path("ARES_TEST_QWEN35_MODEL")) else {
        return;
    };
    let question =
        "Write the line 'the quick brown fox jumps over the lazy dog' four times, numbered.";
    let sampled = |f: &dyn Fn(&mut Sampling)| {
        let mut s = Sampling { temperature: Some(0.8), seed: Some(99), ..Sampling::default() };
        f(&mut s);
        s
    };
    let settings: Vec<(&str, Sampling)> = vec![
        ("nothing set but a temperature", sampled(&|_| {})),
        (
            "penalties",
            sampled(&|s| {
                s.repeat_penalty = Some(1.15);
                s.frequency_penalty = Some(0.2);
                s.presence_penalty = Some(0.1);
                s.repeat_last_n = Some(64);
            }),
        ),
        (
            "dry",
            sampled(&|s| {
                s.dry_multiplier = Some(0.8);
                s.dry_penalty_last_n = Some(128);
            }),
        ),
        (
            "a ban and a bias by text",
            sampled(&|s| {
                s.logit_bias = vec![
                    LogitBias { key: BiasKey::Text(" the".into()), bias: None },
                    LogitBias { key: BiasKey::Text(" fox".into()), bias: Some(2.0) },
                ];
            }),
        ),
        (
            "typical, top-n-sigma, xtc and a dynamic temperature",
            sampled(&|s| {
                s.typical_p = Some(0.9);
                s.top_n_sigma = Some(3.0);
                s.xtc_probability = Some(0.3);
                s.dynatemp_range = Some(0.3);
            }),
        ),
        (
            "an order of its own",
            sampled(&|s| {
                s.top_k = Some(60);
                s.top_p = Some(0.9);
                s.min_p = Some(0.02);
                s.samplers = Some(vec![
                    SamplerKind::Temperature,
                    SamplerKind::MinP,
                    SamplerKind::TopK,
                    SamplerKind::TopP,
                ]);
            }),
        ),
        (
            "the model's own probabilities",
            sampled(&|s| {
                s.top_k = Some(40);
                s.logprobs = Some(LogprobRequest { top: 5, after_chain: false });
            }),
        ),
        (
            "the probabilities after the chain",
            sampled(&|s| {
                s.top_k = Some(40);
                s.top_p = Some(0.9);
                s.logprobs = Some(LogprobRequest { top: 5, after_chain: true });
            }),
        ),
        (
            "greedy with penalties",
            Sampling {
                temperature: Some(0.0),
                repeat_penalty: Some(1.3),
                repeat_last_n: Some(32),
                ..Sampling::default()
            },
        ),
        ("mirostat 2", sampled(&|s| s.mirostat = Some(2))),
        ("mirostat 1", sampled(&|s| s.mirostat = Some(1))),
    ];
    let model = Arc::new(
        Qwen35::new(
            Gguf::open(&path).unwrap(),
            "q",
            &Spread::Single,
            options(Precision::Exact),
            1024,
        )
        .unwrap(),
    );
    let vocab = ares_runtime::Model::vocab(&*model);
    let config =
        RuntimeConfig { pages: 32, cells_per_page: 128, max_slots: 2, ..RuntimeConfig::default() };
    let gpu = GpuConfig::new(&config, &*model);
    let device = Device::open(0, Wait::Block).unwrap();
    let runtime = Runtime::start_gpu(model, vec![device], config, gpu).unwrap();
    for (i, (name, sampling)) in settings.iter().enumerate() {
        let asks = sampling.logprobs.map(|asked| asked.top as usize);
        let (text, probs) =
            ask_with(&runtime, 10 + i as u64, question, sampling.clone(), None, 100);
        eprintln!("RES {name}: {text:?}");
        assert!(text.chars().count() > 20, "{name}: {text:?}");
        check_probabilities(name, &probs, asks);
    }
    let (text, probs) = ask_with(
        &runtime,
        99,
        question,
        sampled(&|s| s.top_k = Some(50)),
        Some(Box::new(Evens(100_000.min(vocab)))),
        60,
    );
    eprintln!("RES a filter: {text:?}");
    assert!(text.chars().count() > 10, "a filter: {text:?}");
    check_probabilities("a filter", &probs, None);
    runtime.shutdown();
}
