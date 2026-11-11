//! The model's own drafts through the whole engine, on the real 27B.
//!
//! Needs two GPUs (`ARES_TEST_CUDA=1`) and the model
//! (`ARES_TEST_QWEN35_27B`); `ARES_TEST_DRAFT_VOCAB` names a draft vocabulary
//! (a token id per line). The model is split by tensor over both devices
//! with exact products, so that a row computes the same in a window of
//! several rows as alone and the texts can be compared:
//!
//! - the text with the model's own drafts is the text without them, for
//!   the most probable token and for sampled tokens (same seed);
//! - drafts are made on the device and a good share of them is accepted.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use ares_api::{
    Engine, Event, EventSink, GenerateRequest, Message, Prompt, RequestId, Role, Sampling,
};
use ares_cuda::{Device, Precision, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{Counters, GpuConfig, Runtime, RuntimeConfig};

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set {name} to run this test");
            None
        }
    }
}

fn ask(runtime: &Runtime, id: u64, question: &str, sampling: Sampling, max_tokens: u32) -> String {
    let (tx, rx) = mpsc::channel();
    let text = Arc::new(Mutex::new(String::new()));
    let sink = {
        let text = text.clone();
        EventSink::new(move |event| {
            match event {
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
                sampling,
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft: true,
                id: RequestId(id),
            },
            sink,
        )
        .unwrap();
    rx.recv().unwrap().unwrap();
    std::mem::take(&mut *text.lock().unwrap())
}

/// The answers to `questions`, each with the most probable token and
/// sampled, with at most `drafts` drafts per window; and the counters.
fn answers(
    model: &Path,
    vocabulary: Option<&Arc<[u32]>>,
    drafts: u32,
    questions: &[&str],
) -> (Vec<String>, Counters) {
    let options = Options {
        precision: Precision::Exact,
        max_rows: 128,
        logit_rows: 16,
        window_rows: if drafts > 0 { drafts + 1 } else { 0 },
        draft_vocabulary: vocabulary.cloned(),
        ..Options::default()
    };
    let model = Arc::new(
        Qwen35::new(
            Gguf::open(model).unwrap(),
            "q",
            &Spread::Tensor(vec![0.625, 0.375]),
            options,
            1024,
        )
        .unwrap(),
    );
    assert_eq!(model.placement().draft.is_some(), drafts > 0);
    let config = RuntimeConfig {
        pages: 20,
        cells_per_page: 128,
        max_slots: 2,
        window_rows: drafts + 1,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let devices =
        vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
    let runtime = Runtime::start_gpu(model, devices, config, gpu).unwrap();
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    let sampled = Sampling {
        temperature: Some(0.8),
        top_k: Some(20),
        top_p: Some(0.95),
        seed: Some(4711),
        ..Sampling::default()
    };
    let mut texts = Vec::new();
    for (i, question) in questions.iter().enumerate() {
        let id = 3 * i as u64;
        texts.push(ask(&runtime, id + 1, question, greedy.clone(), 48));
        texts.push(ask(&runtime, id + 2, question, sampled.clone(), 48));
        // The same seed gives the same text, whatever ran in the slot before
        // (a request with draft windows leaves steps of other sizes behind).
        let again = ask(&runtime, id + 3, question, sampled.clone(), 48);
        assert_eq!(
            texts[2 * i + 1],
            again,
            "a sampled answer is not reproducible ({drafts} drafts)"
        );
    }
    let counters = runtime.snapshot().expect("the engine runs").counters;
    runtime.shutdown();
    (texts, counters)
}

#[test]
fn the_text_with_the_models_own_drafts_is_the_text_without() {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return;
    }
    let Some(model) = env_path("ARES_TEST_QWEN35_27B") else { return };
    if device_count().unwrap() < 2 {
        eprintln!("skipped: the 27B needs two devices");
        return;
    }
    let vocabulary: Option<Arc<[u32]>> = std::env::var_os("ARES_TEST_DRAFT_VOCAB").map(|path| {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.parse().unwrap())
            .collect()
    });
    let questions = [
        "Erkl\u{e4}re in drei S\u{e4}tzen, warum der Himmel blau ist.",
        "Write a Python function that returns the n-th Fibonacci number.",
    ];
    let (plain, _) = answers(&model, vocabulary.as_ref(), 0, &questions);
    let (drafted, counters) = answers(&model, vocabulary.as_ref(), 3, &questions);
    let (tested, accepted) = (counters.drafts_tested_at[1], counters.drafts_accepted_at[1]);
    eprintln!(
        "RES own drafts: {} links in {} steps, {:.2} ms per link; accepted by position {:?} of {:?}; \
         {} tokens in {} steps",
        counters.own_draft_links,
        counters.own_draft_steps,
        counters.own_draft_us as f64 / 1e3 / counters.own_draft_links.max(1) as f64,
        &accepted[..3],
        &tested[..3],
        counters.tokens,
        counters.steps,
    );
    for (i, (a, b)) in plain.iter().zip(&drafted).enumerate() {
        let same = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
        eprintln!(
            "RES answer {i} ({}): {same} of {} characters equal the text without drafts",
            if i % 2 == 0 { "most probable" } else { "sampled" },
            a.chars().count()
        );
    }
    assert_eq!(plain, drafted, "the model's own drafts changed the text");
    assert!(tested[0] >= 50, "the model drafted: {tested:?}");
    assert!(accepted[0] * 3 >= tested[0], "a third of the first drafts is accepted: {accepted:?}");
}
