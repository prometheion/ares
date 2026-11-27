//! Decisions ("Jev mode") in the conductor, against the host executor as an
//! oracle: every question is answered from the logits its prompt gives alone,
//! the state its questions share is computed once and forked into them through
//! the prefix cache, and a decision that ends any other way frees what it held.
//!
//! The synthetic model reads a byte per token, so its tokenizer has no labels
//! for the options ("yes" is three tokens): the tests give the labels
//! ([`Runtime::decide_with`]). The host executor's logit of a label is a fixed
//! function of the token the row picks ([`SyntheticModel::host_score`]), so a
//! fork that starts from the wrong state, or reads the wrong row, answers
//! otherwise.

use std::sync::mpsc;

use ares_api::{
    ChoiceOption, DecisionOutcome, DecisionRequest, DecisionResult, DecisionSink, Engine,
    EngineError, GenerateRequest, Json, Labels, Question, QuestionKind, Sampling, question_prompt,
};
use ares_runtime::Model;
use ares_runtime::testing::Gate;

use super::*;

/// Labels of tokens the synthetic model has: the letters, `y` and `n`, the
/// digits, and four pairs of their own.
fn labels() -> Labels {
    let byte = |c: u8| u32::from(c);
    let mut singles: Vec<u32> = (b'A'..=b'Z').map(byte).collect();
    singles.extend([byte(b'y'), byte(b'n')]);
    singles.extend((b'0'..=b'9').map(byte));
    let pairs = ["AA", "AB", "AC", "AD"].iter().zip(200..).map(|(l, t)| ((*l).into(), t)).collect();
    Labels { singles, pairs }
}

fn model(prefix_cache: Option<u64>) -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.prefix_cache = prefix_cache;
    Arc::new(SyntheticModel::new(config).unwrap())
}

fn config(slots: u32) -> RuntimeConfig {
    RuntimeConfig {
        pages: 512,
        cells_per_page: PAGE,
        max_slots: slots,
        window_rows: 1,
        warm_up: false,
        cache_ram: 64 * CHECKPOINT,
        checkpoint_prompt_interval: PAGE,
        checkpoint_generation_interval: 2 * PAGE,
        ..RuntimeConfig::default()
    }
}

fn string(s: &str) -> Json {
    Json::String(s.into())
}

fn choice(id: &str, keys: &[&str]) -> Question {
    Question {
        id: id.into(),
        instructions: string("Which team should handle this?"),
        kind: QuestionKind::Choice {
            options: keys
                .iter()
                .map(|k| ChoiceOption { key: (*k).into(), description: Json::Null })
                .collect(),
        },
    }
}

fn questions(n: usize) -> Vec<Question> {
    (0..n)
        .map(|i| match i % 3 {
            0 => choice(&format!("route{i}"), &["billing", "shipping", "technical"]),
            1 => Question {
                id: format!("angry{i}"),
                instructions: string("Is the customer angry?"),
                kind: QuestionKind::Noul { when_true: string("shouts"), when_false: Json::Null },
            },
            _ => Question {
                id: format!("urgency{i}"),
                instructions: string("How urgent is it?"),
                kind: QuestionKind::Score {
                    levels: ["later", "soon", "today", "now"].map(string).to_vec(),
                },
            },
        })
        .collect()
}

fn decision(state: &str, questions: Vec<Question>) -> DecisionRequest {
    DecisionRequest {
        model: "synthetic".into(),
        state: string(state),
        questions,
        priority: 0,
        id: fresh_id(),
    }
}

/// Starts `request`; the receiver gets its outcome.
fn start_decision(rt: &Runtime, request: DecisionRequest) -> mpsc::Receiver<DecisionOutcome> {
    let (tx, rx) = mpsc::channel();
    rt.decide_with(request, DecisionSink::new(move |o| tx.send(o).unwrap()), &labels())
        .expect("accepted");
    rx
}

fn outcome(rx: &mpsc::Receiver<DecisionOutcome>) -> DecisionOutcome {
    rx.recv_timeout(LIMIT).expect("the decision ended")
}

fn decided(rt: &Runtime, request: DecisionRequest) -> DecisionResult {
    match outcome(&start_decision(rt, request)) {
        DecisionOutcome::Decided(result) => result,
        other => panic!("not decided: {other:?}"),
    }
}

/// The token the model picks after `prompt`: what the host executor picks at
/// the prompt's last row (its token and the context of every token up to it).
fn pick_after(model: &SyntheticModel, prompt: &[u32]) -> u32 {
    let spec = model.context();
    let mut state = spec.start();
    for i in 0..prompt.len() {
        state = spec.advance(state, &prompt[..=i]);
    }
    let last = prompt.len() as u32 - 1;
    model.host_pick(prompt[last as usize], spec.reduce(state, last)).0
}

/// The answers each question of `request` gets from its own prompt, alone.
fn alone_answers(model: &SyntheticModel, request: &DecisionRequest) -> Vec<ares_api::Answer> {
    let labels = labels();
    request
        .questions
        .iter()
        .map(|q| {
            let rendered = q.render(&labels).unwrap();
            let prompt = model.encode(&question_prompt(&request.state, &rendered.text)).unwrap();
            let pick = pick_after(model, &prompt);
            let logits: Vec<f32> =
                rendered.tokens.iter().map(|t| model.host_score(pick, *t)).collect();
            q.answer(&logits).unwrap()
        })
        .collect()
}

/// The prompts of `request`'s questions, as token lists.
fn prompts_of(model: &SyntheticModel, request: &DecisionRequest) -> Vec<Vec<u32>> {
    let labels = labels();
    request
        .questions
        .iter()
        .map(|q| {
            let text = q.render(&labels).unwrap().text;
            model.encode(&question_prompt(&request.state, &text)).unwrap()
        })
        .collect()
}

fn shared_start(prompts: &[Vec<u32>]) -> usize {
    let shortest = prompts.iter().map(Vec::len).min().unwrap();
    (0..shortest - 1).take_while(|&i| prompts.iter().all(|p| p[i] == prompts[0][i])).count()
}

#[test]
fn every_question_is_answered_as_its_prompt_alone_from_one_computation_of_the_state() {
    let model = model(Some(CHECKPOINT));
    let e = engine(&model, config(8));
    let state = text_of(11, 300);
    let request = decision(&state, questions(3));
    let expected = alone_answers(&model, &request);
    let prompts = prompts_of(&model, &request);
    let shared = shared_start(&prompts);
    let before = e.snapshot();
    let result = decided(&e.rt, request.clone());
    assert_eq!(result.answers, expected);

    // Flide's counts: the shared start once and each question's rest.
    let rest: usize = prompts.iter().map(|p| p.len() - shared).sum();
    assert_eq!(result.prefix_tokens as usize, shared);
    assert_eq!(result.usage.prompt_tokens as usize, shared + rest);
    assert_eq!(result.usage.completion_tokens, 3);
    assert_eq!(result.usage.cached_prompt_tokens, 0, "a cold engine");

    // The state was computed once: the parent's rows and the forks' own rows,
    // each fork restored from the parent's checkpoint.
    let s = e.snapshot();
    let (now, then) = (&s.counters, &before.counters);
    assert!(shared > 300, "the state is most of each prompt: {shared}");
    assert_eq!((now.prefill_rows - then.prefill_rows) as usize, shared + rest, "{s}");
    assert!(now.checkpoints_restored - then.checkpoints_restored >= 3, "{s}");
    assert_eq!(s.counters.decisions, 1);
    assert_eq!(s.counters.requests_finished, 0, "the parts of a decision are not requests");
    assert_eq!(s.pages_free + s.cached_pages, s.pages_capacity, "{s}");
    e.consistent();

    // The answers are those of a generation of each question alone: its first
    // token is the one the logits were read at.
    for (q, answer) in request.questions.iter().zip(&result.answers) {
        let rendered = q.render(&labels()).unwrap();
        let prompt = question_prompt(&request.state, &rendered.text);
        let run = run_one(
            &e.rt,
            GenerateRequest {
                model: "synthetic".into(),
                prompt: prompt.clone(),
                sampling: Sampling::default(),
                max_tokens: Some(1),
                stop: Vec::new(),
                draft: false,
                structure: None,
                priority: 0,
                id: fresh_id(),
            },
        );
        let pick = pick_after(&model, &model.encode(&prompt).unwrap());
        let mut piece = Vec::new();
        model.piece(pick, &mut piece);
        assert_eq!(run.text.as_bytes(), piece, "question {}", answer.question);
    }
    e.consistent();
}

#[test]
fn a_state_the_engine_has_read_costs_the_next_decision_only_its_questions() {
    let model = model(Some(CHECKPOINT));
    let e = engine(&model, config(8));
    let state = text_of(12, 200);
    let first = decided(&e.rt, decision(&state, questions(4)));
    // Other questions about the same state, as an agent's next step asks
    // them: their prompts share what the first ones shared.
    let mut next = questions(4);
    for (q, word) in next.iter_mut().zip(["Does", "Was", "Should", "Would"]) {
        q.instructions = string(&format!("{word} it matter?"));
    }
    let before = e.snapshot();
    let request = decision(&state, next);
    let second = decided(&e.rt, request.clone());
    assert_eq!(second.prefix_tokens, first.prefix_tokens);
    assert_eq!(second.answers, alone_answers(&model, &request));
    // The questions' rows and nothing else: the cache holds the state and the
    // recurrent state at its end, so the forks start from it at once, without
    // a parent, and keep no checkpoint of their own.
    let shared = second.prefix_tokens as u64;
    let rest: u64 = prompts_of(&model, &request).iter().map(|p| p.len() as u64 - shared).sum();
    let s = e.snapshot();
    let computed = s.counters.prefill_rows - before.counters.prefill_rows;
    let replayed = s.counters.replay_rows - before.counters.replay_rows;
    assert_eq!((computed, replayed), (rest, 0), "{s}");
    assert_eq!(s.counters.checkpoints_saved, before.counters.checkpoints_saved, "{s}");
    assert_eq!(s.counters.checkpoints_restored - before.counters.checkpoints_restored, 4, "{s}");
    assert_eq!(second.usage.cached_prompt_tokens as u64, shared);

    // The same decision again: its forks start from the state as well (the
    // first decision's forks left no pages that would hide it).
    let before = e.snapshot();
    let request = decision(&state, questions(4));
    let again = decided(&e.rt, request.clone());
    assert_eq!(again.answers, first.answers);
    let rest: u64 = prompts_of(&model, &request).iter().map(|p| p.len() as u64 - shared).sum();
    let s = e.snapshot();
    let computed = s.counters.prefill_rows - before.counters.prefill_rows;
    let replayed = s.counters.replay_rows - before.counters.replay_rows;
    assert_eq!((computed, replayed), (rest, 0), "{s}");
    e.consistent();
}

#[test]
fn one_question_is_its_own_prompt_and_a_model_without_a_cache_answers_the_same() {
    for prefix_cache in [Some(CHECKPOINT), None] {
        let model = model(prefix_cache);
        let host = HostExecutor::new(model.clone(), model.config().max_rows, 4);
        let host = match prefix_cache {
            Some(_) => host.with_pool(512, PAGE),
            None => host,
        };
        let problems = host.problems();
        let rt = Runtime::start(model.clone(), Box::new(host), config(4)).unwrap();
        for n in [1, 5] {
            let request = decision(&text_of(13, 120), questions(n));
            let result = decided(&rt, request.clone());
            assert_eq!(result.answers, alone_answers(&model, &request), "{prefix_cache:?}, {n}");
        }
        assert!(problems.lock().unwrap().is_empty());
    }
}

#[test]
fn the_forks_of_a_model_with_few_rows_of_its_own_share_steps_within_them() {
    // Flash-Next reads a prompt alone in steps wider than its own rows; forks
    // that arrive together share steps of its own rows.
    let mut shape = SyntheticConfig::small(Split::Single, 1);
    shape.context_length = 2048;
    shape.prefix_cache = Some(CHECKPOINT);
    shape.own_rows = 16;
    let model = Arc::new(SyntheticModel::new(shape).unwrap());
    let e = engine(&model, config(8));
    let request = decision(&text_of(21, 150), questions(6));
    assert_eq!(decided(&e.rt, request.clone()).answers, alone_answers(&model, &request));
    e.consistent();
}

#[test]
fn more_questions_than_slots_take_turns_and_answer_the_same() {
    let model = model(Some(CHECKPOINT));
    let e = engine(&model, config(2));
    let request = decision(&text_of(14, 150), questions(7));
    let result = decided(&e.rt, request.clone());
    assert_eq!(result.answers, alone_answers(&model, &request));
    let s = e.snapshot();
    assert_eq!((s.active, s.pages_free + s.cached_pages), (0, s.pages_capacity), "{s}");
    e.consistent();
}

#[test]
fn decisions_and_generation_run_together_and_neither_changes_the_other() {
    let model = model(Some(CHECKPOINT));
    let e = engine(&model, config(6));
    let prompt = text_of(15, 90);
    let expected_text = fresh_run(|c| c.context_length = 2048, &prompt, 60).text;
    let generation = start(&e.rt, request(&prompt, Some(60))).unwrap();
    let requests: Vec<DecisionRequest> =
        (0..4).map(|i| decision(&text_of(20 + i, 100 + 30 * i as usize), questions(3))).collect();
    let pending: Vec<_> = requests.iter().map(|r| start_decision(&e.rt, r.clone())).collect();
    for (request, rx) in requests.iter().zip(&pending) {
        let DecisionOutcome::Decided(result) = outcome(rx) else { panic!("not decided") };
        assert_eq!(result.answers, alone_answers(&model, request));
    }
    assert_eq!(generation.finish().text, expected_text);
    e.consistent();
}

#[test]
fn a_cancelled_decision_ends_at_once_and_frees_what_it_held() {
    let model = model(Some(CHECKPOINT));
    let gate = Gate::default();
    let e = engine_with(&model, config(4), |host| host.with_gate(gate.clone()));
    let request = decision(&text_of(16, 200), questions(5));
    let id = request.id;
    let rx = start_decision(&e.rt, request);
    e.rt.cancel(id);
    gate.open();
    assert_eq!(outcome(&rx), DecisionOutcome::Cancelled);
    // Nothing of it is left, and the engine goes on.
    let request = decision(&text_of(17, 80), questions(2));
    assert_eq!(decided(&e.rt, request.clone()).answers, alone_answers(&model, &request));
    let s = e.snapshot();
    assert_eq!((s.active, s.pages_free + s.cached_pages), (0, s.pages_capacity), "{s}");
    e.consistent();
}

#[test]
fn a_failing_device_fails_the_decision_with_its_reason() {
    let model = model(Some(CHECKPOINT));
    let e = engine_with(&model, config(4), |host| host.failing_at(2));
    let rx = start_decision(&e.rt, decision(&text_of(18, 100), questions(3)));
    let DecisionOutcome::Failed(EngineError::Internal(why)) = outcome(&rx) else {
        panic!("a failed engine fails its decisions")
    };
    assert!(why.contains("injected"), "{why}");
    let later = start_decision(&e.rt, decision("x", questions(2)));
    assert!(matches!(outcome(&later), DecisionOutcome::Failed(EngineError::Internal(_))));
}

#[test]
fn what_cannot_be_decided_is_refused_before_anything_runs() {
    let model = model(Some(CHECKPOINT));
    let e = engine(&model, config(4));
    let refuse = |request: DecisionRequest| {
        e.rt.decide_with(request, DecisionSink::new(|_| panic!("refused")), &labels()).unwrap_err()
    };
    let mut unknown = decision("s", questions(1));
    unknown.model = "other".into();
    assert_eq!(refuse(unknown), EngineError::ModelNotFound("other".into()));
    let empty = refuse(decision("s", vec![]));
    assert!(matches!(&empty, EngineError::InvalidRequest(m) if m.contains("at least one")));
    // More options than the labels: 26 letters and 4 pairs.
    let keys: Vec<String> = (0..31).map(|i| format!("k{i}")).collect();
    let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
    let wide = refuse(decision("s", vec![choice("wide", &keys)]));
    assert!(matches!(&wide, EngineError::InvalidRequest(m) if m.contains("reads at most 30")));
    // A state longer than the context.
    let long = refuse(decision(&text_of(19, 3000), questions(1)));
    assert!(matches!(long, EngineError::ContextOverflow { .. }), "{long}");
    // Forty questions that each fit, whose prompts are past Flide's limit of
    // 65536 tokens together.
    let long = string(&text_of(20, 1800));
    let many: Vec<Question> = (0..40)
        .map(|i| {
            let mut q = choice(&format!("q{i}"), &["a", "b"]);
            q.instructions = string(&format!("{i}: {}", long.as_str().unwrap()));
            q
        })
        .collect();
    let total = refuse(decision("s", many));
    assert!(
        matches!(&total, EngineError::InvalidRequest(m) if m.contains("limit is 65536")),
        "{total}"
    );
    assert_eq!(e.snapshot().counters.steps, 0, "nothing ran");
}

#[test]
fn a_model_whose_tokenizer_has_no_labels_says_so() {
    let model = model(Some(CHECKPOINT));
    let e = engine(&model, config(4));
    let error =
        e.rt.decide(decision("s", questions(1)), DecisionSink::new(|_| panic!("refused")))
            .unwrap_err();
    assert!(
        matches!(&error, EngineError::Unsupported(m) if m.contains("\"yes\" is not one distinct token")),
        "{error}"
    );
}
