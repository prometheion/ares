//! The model's own drafts through the whole engine, on the real 27B.
//!
//! Needs two GPUs (`ARES_TEST_CUDA=1`) and the model
//! (`ARES_TEST_QWEN38_27B`); `ARES_TEST_DRAFT_VOCAB` names a draft vocabulary
//! (a token id per line). The model is split by tensor over both devices
//! with exact products (the fast kernels are held to the same in
//! `qwen35_windows.rs`, `qwen35_device_runtime.rs` and the server's
//! `tests/real.rs`): a row computes the same in a window of several rows as
//! alone, and the texts can be compared:
//!
//! - the text with the model's own drafts is the text without them, for
//!   the most probable token and for sampled tokens (same seed);
//! - drafts are made on the device and a good share of them is accepted;
//! - the same with the DFlash drafter (`ARES_TEST_DFLASH_27B` names its file),
//!   alone and next to the MTP block.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use ares_api::{
    Engine, Event, EventSink, GenerateRequest, Message, Prompt, RequestId, Role, Sampling,
};
use ares_cuda::{Device, Precision, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{Counters, GpuConfig, Runtime, RuntimeConfig};
use common::helpers::env_path;

mod common;

#[path = "common/kv_env.rs"]
mod kv_env;

fn ask(runtime: &Runtime, id: u64, question: &str, sampling: Sampling, max_tokens: u32) -> String {
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
                    continue_final: false,
                },
                sampling,
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft: true,
                structure: None,
                priority: 0,
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
    answers_with(model, vocabulary, drafts, questions, &Drafting::default())
}

/// The drafters of a run and where the model is.
struct Drafting {
    /// The MTP block takes part, the model file's or that of `mtp_file`.
    mtp: bool,
    mtp_file: Option<PathBuf>,
    dflash: Option<Arc<ares_model::dflash::File>>,
    spread: Spread,
}

impl Default for Drafting {
    fn default() -> Self {
        Self { mtp: true, mtp_file: None, dflash: None, spread: Spread::Tensor(vec![0.625, 0.375]) }
    }
}

/// [`answers`] with the drafters and the placement of `drafting`.
fn answers_with(
    model: &Path,
    vocabulary: Option<&Arc<[u32]>>,
    drafts: u32,
    questions: &[&str],
    drafting: &Drafting,
) -> (Vec<String>, Counters) {
    let options = kv_env::with_env(Options {
        precision: Precision::Exact,
        max_rows: 128,
        logit_rows: 16,
        window_rows: if drafts > 0 { drafts + 1 } else { 0 },
        draft_vocabulary: vocabulary.cloned(),
        mtp: drafting.mtp,
        mtp_file: drafting.mtp_file.as_ref().map(|path| Arc::new(Gguf::open(path).unwrap())),
        dflash: drafting.dflash.clone(),
        ..Options::default()
    });
    let (gguf, tokenizer) = common::opened(model);
    let model =
        Arc::new(Qwen35::new(gguf, tokenizer, "q", &drafting.spread, options, 1024).unwrap());
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
    let Some(model) = env_path("ARES_TEST_QWEN38_27B") else { return };
    if device_count().unwrap() < 2 {
        eprintln!("skipped: the 27B needs two devices");
        return;
    }
    let vocabulary: Option<Arc<[u32]>> = std::env::var_os("ARES_TEST_DRAFT_VOCAB")
        .map(|path| ares_model::files::read_draft_vocabulary(path.as_ref()).unwrap().into());
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
        counters.model_draft_links,
        counters.model_draft_steps,
        counters.model_draft_us as f64 / 1e3 / counters.model_draft_links.max(1) as f64,
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

#[test]
fn the_text_with_dflash_drafts_is_the_text_without() {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return;
    }
    let Some(model) = env_path("ARES_TEST_QWEN38_27B") else { return };
    let Some(file) = env_path("ARES_TEST_DFLASH_27B") else { return };
    if device_count().unwrap() < 2 {
        eprintln!("skipped: the 27B needs two devices");
        return;
    }
    let file = Arc::new(ares_model::dflash::File::open(&file).unwrap());
    let vocabulary: Option<Arc<[u32]>> = std::env::var_os("ARES_TEST_DRAFT_VOCAB")
        .map(|path| ares_model::files::read_draft_vocabulary(path.as_ref()).unwrap().into());
    let questions = [
        "Explain in three sentences why the sky is blue.",
        "Write a Python function that returns the n-th Fibonacci number.",
    ];
    let (plain, _) = answers(&model, vocabulary.as_ref(), 0, &questions);
    for mtp in [false, true] {
        let drafting = Drafting { mtp, dflash: Some(file.clone()), ..Drafting::default() };
        let (drafted, counters) =
            answers_with(&model, vocabulary.as_ref(), 7, &questions, &drafting);
        let (tested, accepted) = (counters.drafts_tested_at[2], counters.drafts_accepted_at[2]);
        eprintln!(
            "RES dflash{}: {} passes, {:.2} ms per drafting step; accepted by position {:?} of \
             {:?}; block accepted {:?} of {:?}; {} tokens in {} steps",
            if mtp { " and MTP block" } else { "" },
            counters.model_draft_steps,
            counters.model_draft_us as f64 / 1e3 / counters.model_draft_steps.max(1) as f64,
            &accepted[..7],
            &tested[..7],
            &counters.drafts_accepted_at[1][..3],
            &counters.drafts_tested_at[1][..3],
            counters.tokens,
            counters.steps,
        );
        for (i, (a, b)) in plain.iter().zip(&drafted).enumerate() {
            let same = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            eprintln!("RES answer {i}: {same} of {} characters equal", a.chars().count());
        }
        assert_eq!(plain, drafted, "the DFlash drafts changed the text");
        if !mtp {
            assert!(tested[0] >= 20, "the DFlash drafter drafted: {tested:?}");
            assert!(accepted[0] * 3 >= tested[0], "a third of the first drafts: {accepted:?}");
        }
    }
}

/// The MTP block drafts under a layer split too (on the last device, which
/// holds the head and the final hidden states), and from a file of its own:
/// `ARES_TEST_QWEN35_MTP_FILE` names a file of architecture `qwen35-mtp` with
/// the 27B's block (made from the model file's own, it drafts what that one
/// drafts). The texts with drafts are the texts without in each placement.
#[test]
fn the_text_with_mtp_drafts_is_the_text_without_under_a_layer_split_and_from_a_file() {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return;
    }
    let Some(model) = env_path("ARES_TEST_QWEN38_27B") else { return };
    if device_count().unwrap() < 2 {
        eprintln!("skipped: the 27B needs two devices");
        return;
    }
    let mtp_file = std::env::var_os("ARES_TEST_QWEN35_MTP_FILE").map(PathBuf::from);
    let questions = [
        "Erkl\u{e4}re in drei S\u{e4}tzen, warum der Himmel blau ist.",
        "Write a Python function that returns the n-th Fibonacci number.",
    ];
    let layer = Spread::Layer(vec![0.5, 0.5]);
    let plain_drafting = Drafting { mtp: false, spread: layer.clone(), ..Drafting::default() };
    let (plain, _) = answers_with(&model, None, 0, &questions, &plain_drafting);
    let mut runs = vec![("the model file's block", None)];
    runs.extend(mtp_file.map(|file| ("the block of a file of its own", Some(file))));
    let mut accepted_of = Vec::new();
    for (what, file) in runs {
        let drafting = Drafting { mtp_file: file, spread: layer.clone(), ..Drafting::default() };
        let (drafted, counters) = answers_with(&model, None, 3, &questions, &drafting);
        let (tested, accepted) = (counters.drafts_tested_at[1], counters.drafts_accepted_at[1]);
        eprintln!(
            "RES layer split, {what}: {} links in {} steps, {:.2} ms per link; accepted by \
             position {:?} of {:?}; {} tokens in {} steps",
            counters.model_draft_links,
            counters.model_draft_steps,
            counters.model_draft_us as f64 / 1e3 / counters.model_draft_links.max(1) as f64,
            &accepted[..3],
            &tested[..3],
            counters.tokens,
            counters.steps,
        );
        assert_eq!(plain, drafted, "the MTP drafts under a layer split changed the text ({what})");
        assert!(tested[0] >= 50, "the model drafted: {tested:?}");
        assert!(accepted[0] * 3 >= tested[0], "a third of the first drafts: {accepted:?}");
        accepted_of.push((tested, accepted));
    }
    if let [own, file] = accepted_of.as_slice() {
        assert_eq!(own, file, "the block from a file drafts what the model file's block drafts");
    }
}
