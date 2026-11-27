//! Decisions ("Jev mode") on the real `qwen35` models through the engine: a
//! request's questions forked from their shared start, each read at the first
//! token of its reply over its labels, must give what each question's prompt
//! gives alone.
//!
//! - The 0.8B (`ARES_TEST_QWEN35_0_8B`) on one GPU: the labels' logits against
//!   the CPU reference run on each question's whole prompt, and the chosen
//!   option against a greedy generation of each question alone.
//! - The 27B (`ARES_TEST_QWEN38_27B`) on two GPUs in a tensor split: the
//!   chosen option against a greedy generation of each question alone, and the
//!   same decision twice (the second from the prefix cache).
//!
//! Needs `ARES_TEST_CUDA=1`. Lines that start with `RES` are the measurements.

use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use ares_api::{
    ChoiceOption, DecisionOutcome, DecisionRequest, DecisionResult, DecisionSink, Engine, Event,
    EventSink, GenerateRequest, Json, Labels, LogprobRequest, Question, QuestionKind, RequestId,
    Sampling, Verdict, question_prompt,
};
use ares_cuda::{Device, Precision, Wait, device_count};
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig};
use common::helpers::env_path;

mod common;

#[path = "common/kv_env.rs"]
mod kv_env;

fn enabled() -> bool {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return false;
    }
    true
}

fn string(s: &str) -> Json {
    Json::String(s.into())
}

/// An email and what is asked about it, as a triage asks: a team, whether
/// the sender is angry, how urgent it is, and a topic out of thirty (which
/// reads the two-letter labels after Z).
fn email(id: u64) -> DecisionRequest {
    let state = Json::Object(vec![
        ("from".into(), string("Erika Mustermann <erika@example.org>")),
        ("subject".into(), string("Zweimal abgebucht!!")),
        (
            "body".into(),
            string(
                "Hallo, ich habe gestern eine Bestellung aufgegeben und sehe heute, dass mir der \
                 Betrag zweimal von meinem Konto abgebucht wurde. Das ist jetzt schon das zweite \
                 Mal in diesem Monat. Ich erwarte, dass das sofort korrigiert wird, sonst kündige \
                 ich mein Konto. Bestellnummer 4711-0815.",
            ),
        ),
    ]);
    let choice = |id: &str, instructions: &str, options: &[(&str, Option<&str>)]| Question {
        id: id.into(),
        instructions: string(instructions),
        kind: QuestionKind::Choice {
            options: options
                .iter()
                .map(|(key, description)| ChoiceOption {
                    key: (*key).into(),
                    description: description.map_or(Json::Null, string),
                })
                .collect(),
        },
    };
    let topics: Vec<String> = (0..30).map(|i| format!("topic {i}")).collect();
    let mut topic_options: Vec<(&str, Option<&str>)> =
        topics.iter().map(|t| (t.as_str(), None)).collect();
    topic_options[17] = ("double charge", Some("money taken twice for one order"));
    let questions = vec![
        choice(
            "route",
            "Which team should handle this email?",
            &[
                ("billing", Some("payments, refunds, invoices")),
                ("shipping", Some("parcels and delivery")),
                ("technical", Some("the website and the app")),
                ("sales", None),
            ],
        ),
        Question {
            id: "angry".into(),
            instructions: string("Is the sender angry?"),
            kind: QuestionKind::Noul {
                when_true: string("the tone is upset"),
                when_false: Json::Null,
            },
        },
        Question {
            id: "urgency".into(),
            instructions: string("How urgent is this email?"),
            kind: QuestionKind::Score {
                levels: ["can wait", "this week", "today", "within the hour", "right now"]
                    .map(string)
                    .to_vec(),
            },
        },
        choice("topic", "What is the email about?", &topic_options),
    ];
    DecisionRequest { model: "q".into(), state, questions, priority: 0, id: RequestId(id) }
}

fn decide(runtime: &Runtime, request: DecisionRequest) -> DecisionResult {
    let (tx, rx) = mpsc::channel();
    runtime.decide(request, DecisionSink::new(move |o| tx.send(o).unwrap())).unwrap();
    match rx.recv().unwrap() {
        DecisionOutcome::Decided(result) => result,
        other => panic!("not decided: {other:?}"),
    }
}

/// The token a greedy generation of `question` alone starts its reply with.
fn first_token(runtime: &Runtime, id: u64, state: &Json, question: &str) -> u32 {
    let (tx, rx) = mpsc::channel();
    let token = Arc::new(Mutex::new(None));
    let seen = token.clone();
    let sink = EventSink::new(move |event| {
        match &event {
            Event::Logprobs(probs) => {
                let mut seen = seen.lock().unwrap();
                if seen.is_none() {
                    *seen = probs.first().map(|p| p.token);
                }
            }
            Event::Done { .. } => {
                let _ = tx.send(Ok(()));
            }
            Event::Error(e) => {
                let _ = tx.send(Err(e.clone()));
            }
            Event::Text(_) => {}
        }
        true
    });
    runtime
        .submit(
            GenerateRequest {
                model: "q".into(),
                prompt: question_prompt(state, question),
                sampling: Sampling {
                    temperature: Some(0.0),
                    logprobs: Some(LogprobRequest { top: 0, after_chain: false }),
                    ..Sampling::default()
                },
                max_tokens: Some(1),
                stop: Vec::new(),
                draft: false,
                structure: None,
                priority: 0,
                id: RequestId(id),
            },
            sink,
        )
        .unwrap();
    rx.recv().unwrap().unwrap();
    token.lock().unwrap().expect("the reply's first token")
}

/// The option a verdict chose, as the index of its label.
fn chosen(question: &Question, verdict: &Verdict) -> usize {
    match (&question.kind, verdict) {
        (QuestionKind::Choice { options }, Verdict::Choice { choice, .. }) => {
            options.iter().position(|o| &o.key == choice).unwrap()
        }
        (_, Verdict::Score { probabilities, .. }) => first_max(probabilities),
        (_, Verdict::Noul { probability }) => usize::from(*probability < 0.5),
        _ => panic!("a verdict of the wrong kind"),
    }
}

fn first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

fn probabilities(verdict: &Verdict) -> Vec<f64> {
    match verdict {
        Verdict::Choice { probabilities, .. } => probabilities.iter().map(|(_, p)| *p).collect(),
        Verdict::Score { probabilities, .. } => probabilities.clone(),
        Verdict::Noul { probability } => vec![*probability, 1.0 - probability],
    }
}

/// Checks the chosen options of `result` against a greedy generation of each
/// question alone: where the generation starts with a label of the question,
/// it is the chosen one. Returns how many questions it checked so.
fn against_greedy(
    runtime: &Runtime,
    labels: &Labels,
    request: &DecisionRequest,
    result: &DecisionResult,
) -> usize {
    let mut checked = 0;
    for (i, (question, answer)) in request.questions.iter().zip(&result.answers).enumerate() {
        let rendered = question.render(labels).unwrap();
        let token = first_token(runtime, 1000 + i as u64, &request.state, &rendered.text);
        let at = rendered.tokens.iter().position(|t| *t == token);
        eprintln!(
            "RES question {}: chose option {} (p {:?}); greedy alone starts with token {token} \
             (label {at:?})",
            question.id,
            chosen(question, &answer.verdict),
            probabilities(&answer.verdict)
        );
        if let Some(at) = at {
            assert_eq!(at, chosen(question, &answer.verdict), "question {}", question.id);
            checked += 1;
        }
    }
    checked
}

#[test]
fn the_small_model_decides_as_the_reference_reads_each_question_alone() {
    let (true, Some(path)) = (enabled(), env_path("ARES_TEST_QWEN35_0_8B")) else {
        return;
    };
    let options = kv_env::with_env(Options {
        precision: Precision::Exact,
        max_rows: 256,
        logit_rows: 16,
        ..Options::default()
    });
    let (gguf, tokenizer) = common::opened(&path);
    let model =
        Arc::new(Qwen35::new(gguf, tokenizer, "q", &Spread::Single, options, 2048).unwrap());
    let config = RuntimeConfig {
        pages: 64,
        cells_per_page: 128,
        max_slots: 4,
        cache_ram: 256 << 20,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let device = Device::open(0, Wait::Block).unwrap();
    let runtime = Runtime::start_gpu(model.clone(), vec![device], config, gpu).unwrap();
    let labels = runtime.labels().unwrap();
    eprintln!("RES labels: {} pairs after Z", labels.pairs.len());
    assert_eq!(labels.most(), 256, "Qwen's vocabulary has the two-letter labels");

    let request = email(1);
    let begun = Instant::now();
    let result = decide(&runtime, request.clone());
    eprintln!(
        "RES 0.8B decision of {} questions: {:.1} ms, {} tokens ({} shared)",
        request.questions.len(),
        begun.elapsed().as_secs_f64() * 1e3,
        result.usage.prompt_tokens,
        result.prefix_tokens
    );

    // The same decision again, before anything else takes the cache's tails:
    // its shared start from the prefix cache, the same answers.
    let again = decide(&runtime, email(2));
    assert!(again.usage.cached_prompt_tokens > 0, "{again:?}");
    for (a, b) in result.answers.iter().zip(&again.answers) {
        let diff = probabilities(&a.verdict)
            .iter()
            .zip(probabilities(&b.verdict))
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(diff < 0.01, "{a:?} against {b:?}");
    }
    // The CPU reference on each question's whole prompt.
    let gguf = ares_gguf::Gguf::open(&path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut worst = 0.0f64;
    for (question, answer) in request.questions.iter().zip(&result.answers) {
        let rendered = question.render(&labels).unwrap();
        let prompt = model.encode(&question_prompt(&request.state, &rendered.text)).unwrap();
        let mut state = SequenceState::new(&c);
        let window = reference.forward(&mut state, &prompt, LogitRows::Last).unwrap();
        let logits = window.logits_of(prompt.len() - 1).unwrap();
        let read: Vec<f32> = rendered.tokens.iter().map(|t| logits[*t as usize]).collect();
        let want = question.answer(&read).unwrap();
        let (got, wanted) = (probabilities(&answer.verdict), probabilities(&want.verdict));
        let diff = got.iter().zip(&wanted).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(diff);
        eprintln!("RES question {}: reference {wanted:?}, engine {got:?}", question.id);
        let mut sorted = wanted.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let margin = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
        if margin > 0.05 {
            assert_eq!(
                chosen(question, &answer.verdict),
                chosen(question, &want.verdict),
                "question {}",
                question.id
            );
        }
        assert!(diff < 0.03, "question {}: {got:?} against {wanted:?}", question.id);
    }
    eprintln!("RES the largest difference of a probability from the reference: {worst:.4}");
    let checked = against_greedy(&runtime, &labels, &request, &result);
    eprintln!("RES {checked} of {} questions start with a label alone", request.questions.len());

    eprintln!("RES counters: {:?}", runtime.snapshot().unwrap().counters);
}

#[test]
fn the_27b_decides_as_each_question_alone_starts_its_reply() {
    if !enabled() {
        return;
    }
    let Some(path) = env_path("ARES_TEST_QWEN38_27B") else { return };
    if device_count().unwrap() < 2 {
        eprintln!("skipped: the 27B needs two devices");
        return;
    }
    let runtime = runtime_27b(&path);
    let labels = runtime.labels().unwrap();
    let request = email(1);
    let begun = Instant::now();
    let result = decide(&runtime, request.clone());
    eprintln!(
        "RES 27B decision of {} questions: {:.1} ms, {} tokens ({} shared)",
        request.questions.len(),
        begun.elapsed().as_secs_f64() * 1e3,
        result.usage.prompt_tokens,
        result.prefix_tokens
    );
    let begun = Instant::now();
    let again = decide(&runtime, email(2));
    eprintln!(
        "RES 27B the same decision again: {:.1} ms, {} of the shared {} from the cache",
        begun.elapsed().as_secs_f64() * 1e3,
        again.usage.cached_prompt_tokens,
        again.prefix_tokens
    );
    for (question, (a, b)) in
        request.questions.iter().zip(result.answers.iter().zip(&again.answers))
    {
        assert_eq!(chosen(question, &a.verdict), chosen(question, &b.verdict));
    }
    let checked = against_greedy(&runtime, &labels, &request, &result);
    assert!(checked >= 2, "the 27B answers most of these with a label: {checked}");
}

/// The 27B on both cards as the owner runs it: a tensor split of 0.625 and
/// 0.375.
fn runtime_27b(path: &Path) -> Runtime {
    let options = kv_env::with_env(Options {
        precision: Precision::Exact,
        max_rows: 256,
        logit_rows: 16,
        ..Options::default()
    });
    let (gguf, tokenizer) = common::opened(path);
    let model = Arc::new(
        Qwen35::new(gguf, tokenizer, "q", &Spread::Tensor(vec![0.625, 0.375]), options, 4096)
            .unwrap(),
    );
    let config = RuntimeConfig {
        pages: 32,
        cells_per_page: 128,
        max_slots: 4,
        cache_ram: 1 << 30,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let devices =
        vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
    Runtime::start_gpu(model, devices, config, gpu).unwrap()
}
