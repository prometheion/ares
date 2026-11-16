//! What a request can ask of the sampler, through the server, with the real
//! Qwen3.5-0.8B on a GPU, and with the 27B and its draft block for what drafts
//! do to it.
//!
//! The server runs in this process over the real engine (requests go through
//! the router without a socket), so that a test can read the tokens the
//! engine generated with their probabilities, count the drafts, and give the
//! engine a rule about the tokens of a reply, which is the hook the grammars
//! will use. `real.rs` runs the binary over sockets.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL` (the 0.8B);
//! `ARES_TEST_DEVICE` names the device (default 0). The 0.8B has no draft
//! block, so nothing is drafted for it: the tests of drafts need
//! `ARES_TEST_QWEN35_27B` (Qwen3.8-27B-UD-Q4_K_S.gguf) and two GPUs, and
//! compare the reply to a request with the reply to the same request that says
//! `"draft": false`. Lines that start with `RES` are measurements.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     ARES_TEST_QWEN35_27B=Qwen3.8-27B-UD-Q4_K_S.gguf \
//!     cargo test --release -p ares-server --test real_sampling -- --test-threads=1
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ares_api::{
    ChatFormat, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, EventSink,
    GenerateRequest, ModelCard, Prompt, RequestId,
};
use ares_cuda::{Device, Precision, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig, TokenFilter};
use ares_sampler::TokenMask;
use ares_tokenizer::{ChatOptions, Json};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, header};
use serde_json::{Value, json};
use tower::ServiceExt;

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";

/// A text that is repeated, so that penalties have something to do and drafts
/// something to find.
const QUESTION: &str =
    "Write the line 'the quick brown fox jumps over the lazy dog' four times, numbered.";
const CONTINUE: &str = "the quick brown fox jumps over the lazy dog. the quick brown fox jumps \
                        over the lazy dog. the quick brown";

fn model_path() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    match std::env::var_os("ARES_TEST_QWEN35_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to the Qwen3.5-0.8B model file");
            None
        }
    }
}

/// The 27B, whose draft block gives the drafts the tests compare with none.
fn drafting_model() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    match std::env::var_os("ARES_TEST_QWEN35_27B") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_27B to the 27B Q4_K_S file (two GPUs)");
            None
        }
    }
}

/// The rules the mask test switches between.
const NONE: usize = 0;
const BY_PLACE: usize = 1;
const WITHOUT_MOST: usize = 2;

/// A rule about the tokens a reply may be made of: which tokens may follow
/// the reply so far, by the number of tokens it has.
type Rule = Arc<dyn Fn(usize, &mut TokenMask) + Send + Sync>;

/// The engine, with every request restricted by `rule`.
struct Restricted {
    inner: Arc<Runtime>,
    rule: Rule,
}

struct Follow(Rule);

impl TokenFilter for Follow {
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
        (self.0)(reply.len() + drafts.len(), mask);
        true
    }
}

impl Engine for Restricted {
    fn models(&self) -> Vec<ModelCard> {
        self.inner.models()
    }

    fn chat_format(&self, model: &str) -> Option<ChatFormat> {
        self.inner.chat_format(model)
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        self.inner.submit_filtered(request, events, Box::new(Follow(self.rule.clone())))
    }

    fn decide(&self, request: DecisionRequest, outcome: DecisionSink) -> Result<(), EngineError> {
        self.inner.decide(request, outcome)
    }

    fn cancel(&self, id: RequestId) {
        self.inner.cancel(id);
    }

    fn stats(&self) -> Option<EngineStats> {
        self.inner.stats()
    }
}

/// The server over the engine.
struct Served {
    app: Router,
    runtime: Arc<Runtime>,
    model: Arc<Qwen35>,
}

impl Served {
    /// Starts the engine on the 0.8B, which drafts nothing, with `rule` about
    /// the tokens of every reply if there is one.
    fn start(path: &PathBuf, rule: Option<Rule>) -> Self {
        let options = Options {
            precision: Precision::Exact,
            max_rows: 128,
            logit_rows: 16,
            window_rows: 8,
            pipeline_rows: 32,
            ..Options::default()
        };
        let model =
            Qwen35::new(Gguf::open(path).unwrap(), "q", &Spread::Single, options, 4096).unwrap();
        let config = RuntimeConfig {
            pages: 96,
            cells_per_page: 128,
            max_slots: 4,
            ..RuntimeConfig::default()
        };
        let ordinal =
            std::env::var("ARES_TEST_DEVICE").ok().and_then(|d| d.parse().ok()).unwrap_or(0);
        Self::serve(model, config, vec![Device::open(ordinal, Wait::Block).unwrap()], rule)
    }

    /// Starts the engine on the 27B over both cards with its draft block and
    /// the n-gram drafter, with exact products so that a row computes
    /// the same in a window as alone.
    fn start_drafting(path: &PathBuf, rule: Option<Rule>) -> Self {
        let options = Options {
            precision: Precision::Exact,
            max_rows: 128,
            logit_rows: 16,
            window_rows: 4,
            ..Options::default()
        };
        let spread = Spread::Tensor(vec![0.625, 0.375]);
        let model = Qwen35::new(Gguf::open(path).unwrap(), "q", &spread, options, 1024).unwrap();
        let config = RuntimeConfig {
            pages: 24,
            cells_per_page: 128,
            max_slots: 4,
            window_rows: 4,
            ..RuntimeConfig::default()
        };
        let devices =
            vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
        Self::serve(model, config, devices, rule)
    }

    fn serve(
        model: Qwen35,
        config: RuntimeConfig,
        devices: Vec<Device>,
        rule: Option<Rule>,
    ) -> Self {
        let chat = ChatOptions {
            variables: vec![("enable_thinking".to_owned(), Json::Bool(false))],
            ..ChatOptions::default()
        };
        let model = Arc::new(model.with_chat_options(chat));
        let as_model: Arc<dyn Model> = model.clone();
        let gpu = GpuConfig::new(&config, &*as_model);
        let runtime = Arc::new(Runtime::start_gpu(as_model, devices, config, gpu).unwrap());
        let engine: Arc<dyn Engine> = match rule {
            Some(rule) => Arc::new(Restricted { inner: runtime.clone(), rule }),
            None => runtime.clone(),
        };
        Self { app: ares_server::router(engine), runtime, model }
    }

    async fn post(&self, path: &str, body: &Value) -> (u16, String) {
        let request = Request::post(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = self.app.clone().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    /// The answer to a request that is to succeed, as the JSON of the server.
    async fn answer(&self, path: &str, body: &Value) -> Value {
        let (status, text) = self.post(path, body).await;
        assert_eq!(status, 200, "{text}");
        serde_json::from_str(&text).unwrap()
    }

    /// The first completion of a chat request with `extra` set on top of a
    /// greedy one that asks for the probabilities.
    async fn chat(&self, extra: Value) -> Said {
        let value = self.answer(CHAT, &chat_body(extra)).await;
        said(&value["choices"][0])
    }

    /// The tokens the engine has drafted and verified, and those that were right.
    fn drafts(&self) -> (u64, u64) {
        let counters = self.runtime.snapshot().expect("the engine runs").counters;
        (counters.draft_rows, counters.drafts_accepted)
    }

    fn prompt_ids(&self, text: &str) -> Vec<u32> {
        Model::encode(&*self.model, &Prompt::Text(text.to_owned())).unwrap()
    }
}

/// A chat request: greedy, 48 tokens at most, with the probabilities of the
/// tokens, and `extra` on top.
fn chat_body(extra: Value) -> Value {
    let mut body = json!({
        "model": "q",
        "messages": [{"role": "user", "content": QUESTION}],
        "temperature": 0,
        "max_tokens": 48,
        "logprobs": true,
        "top_logprobs": 0,
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

/// What a completion said: its text and, token by token, what the server
/// reported of the tokens.
#[derive(Debug, Default, Clone)]
struct Said {
    text: String,
    ids: Vec<u64>,
    pieces: Vec<String>,
    bytes: Vec<u8>,
    /// The `logprob` of each token, or its `prob` after the chain.
    scores: Vec<f64>,
    /// The alternatives of each token with their scores, best first.
    alternatives: Vec<Vec<(u64, f64)>>,
    finish: String,
}

fn said(choice: &Value) -> Said {
    let text = choice["message"]["content"]
        .as_str()
        .or_else(|| choice["text"].as_str())
        .unwrap_or_default()
        .to_owned();
    let mut said = Said {
        text,
        finish: choice["finish_reason"].as_str().unwrap_or_default().to_owned(),
        ..Said::default()
    };
    let score = |entry: &Value| entry["logprob"].as_f64().or_else(|| entry["prob"].as_f64());
    for token in choice["logprobs"]["content"].as_array().into_iter().flatten() {
        said.ids.push(token["id"].as_u64().unwrap());
        said.pieces.push(token["token"].as_str().unwrap().to_owned());
        said.bytes
            .extend(token["bytes"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8));
        said.scores.push(score(token).unwrap());
        let list = token.get("top_logprobs").or_else(|| token.get("top_probs"));
        said.alternatives.push(
            list.and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|a| (a["id"].as_u64().unwrap(), score(a).unwrap()))
                .collect(),
        );
    }
    said
}

/// The characters two texts have in common from the start.
fn common(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Whether a token of `ids` is among the `window` before it.
fn repeats_within(ids: &[u64], window: usize) -> bool {
    (0..ids.len()).any(|i| ids[i.saturating_sub(window)..i].contains(&ids[i]))
}

/// Characters of a token the DRY sampler of llama.cpp ends a sequence at
/// (its default sequence breakers).
fn breaks(piece: &str) -> bool {
    piece.contains(['\n', ':', '"', '*'])
}

// ---------------------------------------------------------------------------
// Penalties, DRY and the logit bias

/// Presence and frequency are subtracted from the logit of every token in the
/// window, by whether it is there and by how often. The model's own
/// probabilities do not move with them, so the pick of every step can be
/// worked out from the probabilities the server reports: the best of the
/// alternatives after the penalties.
#[tokio::test]
async fn presence_and_frequency_penalties_pick_what_their_definition_says() {
    let Some(path) = model_path() else { return };
    let served = Served::start(&path, None);
    let prompt = served.prompt_ids(CONTINUE);
    let window = 8usize;
    let (presence, frequency) = (2.0f64, 1.5f64);
    let body = |extra: Value| {
        let mut body = json!({
            "model": "q", "prompt": CONTINUE, "temperature": 0, "max_tokens": 40, "n_probs": 20,
        });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        body
    };
    let ask = |extra: Value| {
        let served = &served;
        let body = body(extra);
        async move {
            let value = served.answer(COMPLETIONS, &body).await;
            said(&value["choices"][0])
        }
    };
    let plain = ask(json!({})).await;
    let penalised = ask(json!({
        "presence_penalty": presence, "frequency_penalty": frequency, "repeat_last_n": window,
    }))
    .await;
    assert_ne!(plain.ids, penalised.ids, "the penalties change nothing");
    // Without a penalty the best of the alternatives is the pick; with them
    // it is the best after the window's counts are taken off.
    let mut changed = 0;
    let mut history: Vec<u64> = prompt.iter().map(|t| u64::from(*t)).collect();
    for (i, token) in penalised.ids.iter().enumerate() {
        let from = history.len().saturating_sub(window);
        let mut counts: HashMap<u64, u32> = HashMap::new();
        for seen in &history[from..] {
            *counts.entry(*seen).or_default() += 1;
        }
        let score = |(id, logprob): &(u64, f64)| {
            let count = counts.get(id).copied().unwrap_or(0);
            logprob - if count > 0 { presence + frequency * f64::from(count) } else { 0.0 }
        };
        let alternatives = &penalised.alternatives[i];
        let best = alternatives.iter().map(score).fold(f64::NEG_INFINITY, f64::max);
        let weakest = alternatives.last().expect("20 alternatives").1;
        // A token outside the 20 alternatives is at most as probable as the
        // last of them and is not penalised less than none.
        assert!(best >= weakest, "step {i}: the pick may be outside the alternatives");
        let picked = alternatives
            .iter()
            .find(|(id, _)| id == token)
            .unwrap_or_else(|| panic!("step {i}: token {token} is not among the alternatives"));
        assert!(
            score(picked) >= best - 1e-3,
            "step {i}: picked {token} at {} but {best} was there",
            score(picked)
        );
        changed += usize::from(*token != alternatives[0].0);
        history.push(*token);
    }
    eprintln!(
        "RES penalties: {changed} of {} picks are not the most probable",
        penalised.ids.len()
    );
    assert!(changed >= 1, "no pick was moved by a penalty");
    served.runtime.shutdown();
}

/// A repetition penalty that is large lowers the logit of every token in the
/// window so far that the best token of the rest is picked: nothing repeats
/// within the window, and with a window of 0 the penalty is off.
#[tokio::test]
async fn a_repetition_penalty_keeps_the_window_from_repeating_and_a_window_of_0_turns_it_off() {
    let Some(path) = model_path() else { return };
    let served = Served::start(&path, None);
    let window = 8;
    // The reply does not end early, so that the whole window is looked at.
    let plain = served.chat(json!({"ignore_eos": true})).await;
    assert!(repeats_within(&plain.ids, window), "nothing to avoid: {:?}", plain.pieces);
    for extra in [
        json!({"repeat_penalty": 100.0, "repeat_last_n": window, "ignore_eos": true}),
        json!({"repetition_penalty": 100.0, "repeat_last_n": window, "ignore_eos": true}),
    ] {
        let said = served.chat(extra.clone()).await;
        assert!(said.ids.len() >= 16, "{extra}: {:?}", said.text);
        assert!(!repeats_within(&said.ids, window), "{extra}: {:?}", said.pieces);
    }
    // The window is the last tokens of the prompt and the reply: a token
    // further back is free to come again.
    let narrow =
        served.chat(json!({"repeat_penalty": 100.0, "repeat_last_n": 4, "ignore_eos": true})).await;
    assert!(!repeats_within(&narrow.ids, 4), "{:?}", narrow.pieces);
    assert!(repeats_within(&narrow.ids, window), "a token far back is free: {:?}", narrow.pieces);
    let off =
        served.chat(json!({"repeat_penalty": 100.0, "repeat_last_n": 0, "ignore_eos": true})).await;
    assert_eq!(off.ids, plain.ids, "a window of 0 is no penalty");
    served.runtime.shutdown();
}

/// DRY lowers a token that would repeat a sequence of tokens the reply has
/// already had, by the length of the sequence: a large multiplier leaves no
/// sequence of three tokens (more than the allowed length of 2) that appears
/// twice, apart from those the sequence breakers cut.
#[tokio::test]
async fn dry_leaves_no_sequence_longer_than_the_allowed_length_repeated() {
    let Some(path) = model_path() else { return };
    let served = Served::start(&path, None);
    let repeated = |said: &Said| -> usize {
        let triples: Vec<_> = (2..said.ids.len())
            .filter(|i| !said.pieces[i - 2..=*i].iter().any(|p| breaks(p)))
            .map(|i| &said.ids[i - 2..=i])
            .collect();
        (0..triples.len()).filter(|i| triples[..*i].contains(&triples[*i])).count()
    };
    let plain = served.chat(json!({"ignore_eos": true})).await;
    let with = served
        .chat(json!({
            "dry_multiplier": 100.0, "dry_base": 1.75, "dry_allowed_length": 2,
            "dry_penalty_last_n": 64, "ignore_eos": true,
        }))
        .await;
    eprintln!(
        "RES dry: {} repeated triples without, {} with it",
        repeated(&plain),
        repeated(&with)
    );
    assert!(repeated(&plain) >= 3, "nothing to avoid: {:?}", plain.pieces);
    assert!(with.ids.len() >= 16, "{:?}", with.text);
    assert_eq!(repeated(&with), 0, "{:?}", with.pieces);
    served.runtime.shutdown();
}

/// A bias is added to the logit of a token: a large one makes it the pick
/// every time, one that is very negative bans it, `false` bans it, and a text
/// gives its bias to every token that makes it up. `ignore_eos` bans the end.
#[tokio::test]
async fn a_logit_bias_forces_and_bans_tokens_and_ignore_eos_bans_the_end() {
    let Some(path) = model_path() else { return };
    let served = Served::start(&path, None);
    let plain = served.chat(json!({"top_logprobs": 5, "max_tokens": 12})).await;
    let (first, second) = (plain.alternatives[0][0].0, plain.alternatives[0][1].0);
    assert_eq!(first, plain.ids[0]);

    // The second token of the first step becomes the pick of every step, in
    // the OpenAI form (a map of ids) and in llama.cpp's (pairs).
    for bias in [json!({ second.to_string(): 100 }), json!([[second, 100]])] {
        let forced = served.chat(json!({"logit_bias": bias, "max_tokens": 8})).await;
        assert_eq!(forced.ids, vec![second; 8], "{bias}");
    }
    // Banned in three ways, it is never the pick.
    for bias in [json!({ first.to_string(): -100 }), json!([[first, false]])] {
        let banned = served.chat(json!({"logit_bias": bias})).await;
        assert!(banned.ids.len() >= 8 && !banned.ids.contains(&first), "{bias}");
        assert_eq!(banned.ids[0], second, "the best of the rest is picked: {bias}");
    }
    // A text gives the bias to every token it is made of: the reply is made
    // of nothing else.
    let by_text = served.chat(json!({"logit_bias": [["fox", 100]], "max_tokens": 6})).await;
    assert!(
        !by_text.text.is_empty() && by_text.text.chars().all(|c| "fox".contains(c)),
        "{:?}",
        by_text.text
    );

    // The answer ends where the model ends it; with `ignore_eos` it does not.
    let short = served
        .chat(json!({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 64}))
        .await;
    assert_eq!(short.finish, "stop");
    assert!(short.ids.len() < 40, "{:?}", short.text);
    let going = served
        .chat(json!({
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 64,
            "ignore_eos": true,
        }))
        .await;
    assert_eq!(going.finish, "length", "{:?}", going.text);
    assert_eq!(going.ids.len(), 64, "{:?}", going.pieces);
    served.runtime.shutdown();
}

// ---------------------------------------------------------------------------
// The probabilities

#[tokio::test]
async fn the_probabilities_of_the_tokens_come_back_in_every_form_and_add_up() {
    let Some(path) = model_path() else { return };
    let served = Served::start(&path, None);

    // OpenAI's chat form, the model's own: of a greedy token the most
    // probable, in order, adding to no more than 1, and the bytes of the
    // tokens are the bytes of the text.
    let greedy = served.chat(json!({"top_logprobs": 20})).await;
    assert!(greedy.ids.len() >= 16);
    for (i, alternatives) in greedy.alternatives.iter().enumerate() {
        assert!((10..=20).contains(&alternatives.len()), "step {i}: {}", alternatives.len());
        assert_eq!(alternatives[0], (greedy.ids[i], greedy.scores[i]), "step {i}");
        assert!(alternatives.windows(2).all(|w| w[0].1 >= w[1].1), "step {i}: not in order");
        assert!(alternatives[0].1 <= 0.0);
        let total: f64 = alternatives.iter().map(|a| a.1.exp()).sum();
        assert!(total <= 1.0 + 1e-4, "step {i}: the probabilities add to {total}");
    }
    let sure = greedy.alternatives.iter().filter(|a| a[0].1.exp() > 0.95).count();
    assert!(sure >= 4, "a repeated line has tokens the model is sure of: {sure}");
    assert_eq!(String::from_utf8_lossy(&greedy.bytes), greedy.text);

    // With a temperature the pick is not always the best, and its probability
    // is in the list when it is among the best.
    let sampled = served
        .chat(json!({"temperature": 1.5, "seed": 3, "top_logprobs": 20, "max_tokens": 40}))
        .await;
    for (i, alternatives) in sampled.alternatives.iter().enumerate() {
        match alternatives.iter().find(|a| a.0 == sampled.ids[i]) {
            Some(a) => assert_eq!(a.1, sampled.scores[i], "step {i}"),
            None => assert!(sampled.scores[i] <= alternatives.last().unwrap().1, "step {i}"),
        }
    }
    assert!(sampled.ids.iter().zip(&sampled.alternatives).any(|(id, a)| *id != a[0].0));

    // llama.cpp's form after the chain: the probabilities of the tokens the
    // chain left, at the temperature it ended with. Two tokens are left, and
    // they add to 1.
    let post = served
        .chat(json!({
            "temperature": 1.0, "top_k": 2, "seed": 4, "n_probs": 5, "post_sampling_probs": true,
            "max_tokens": 30,
        }))
        .await;
    assert!(post.ids.len() >= 16);
    for (i, alternatives) in post.alternatives.iter().enumerate() {
        assert!(alternatives.len() <= 2, "step {i}: {alternatives:?}");
        let total: f64 = alternatives.iter().map(|a| a.1).sum();
        assert!((total - 1.0).abs() < 1e-3, "step {i}: the probabilities add to {total}");
        let own = alternatives.iter().find(|a| a.0 == post.ids[i]).expect("the pick is left");
        assert_eq!(own.1, post.scores[i]);
    }

    // The older form of the completions endpoint: the pieces make the text
    // and the offsets say where each begins.
    let value = served
        .answer(
            COMPLETIONS,
            &json!({"model": "q", "prompt": CONTINUE, "temperature": 0, "max_tokens": 24,
                    "logprobs": 3}),
        )
        .await;
    let choice = &value["choices"][0];
    let logprobs = &choice["logprobs"];
    let tokens: Vec<&str> =
        logprobs["tokens"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
    assert_eq!(tokens.concat(), choice["text"].as_str().unwrap());
    let offsets: Vec<usize> = logprobs["text_offset"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_u64().unwrap() as usize)
        .collect();
    let mut at = 0;
    for (token, offset) in tokens.iter().zip(&offsets) {
        assert_eq!(*offset, at);
        at += token.chars().count();
    }
    assert_eq!(logprobs["token_logprobs"].as_array().unwrap().len(), tokens.len());
    assert_eq!(logprobs["top_logprobs"].as_array().unwrap().len(), tokens.len());
    assert!(logprobs["top_logprobs"][0].as_object().unwrap().len() <= 3);

    // Streamed, the same request gives the same tokens and probabilities.
    let streamed = served.post(CHAT, &chat_body(json!({"stream": true, "top_logprobs": 3}))).await;
    assert_eq!(streamed.0, 200, "{}", streamed.1);
    let chunks: Vec<Value> = streamed
        .1
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let (mut ids, mut scores) = (Vec::new(), Vec::new());
    let mut text = String::new();
    for chunk in &chunks {
        let choice = &chunk["choices"][0];
        text.push_str(choice["delta"]["content"].as_str().unwrap_or_default());
        for token in choice["logprobs"]["content"].as_array().into_iter().flatten() {
            ids.push(token["id"].as_u64().unwrap());
            scores.push(token["logprob"].as_f64().unwrap());
        }
    }
    let plain = served.chat(json!({"top_logprobs": 3})).await;
    assert_eq!(ids, plain.ids);
    assert_eq!(scores, plain.scores);
    assert_eq!(text, plain.text);
    served.runtime.shutdown();
}

// ---------------------------------------------------------------------------
// The mask

/// A rule about the tokens of a reply is held by every row: with drafts that
/// are checked against rows of a window, and without them, a token of the
/// reply is one the rule allowed at its place.
#[tokio::test]
async fn a_mask_restricts_every_row_of_a_reply_with_drafts_and_without() {
    let Some(path) = drafting_model() else { return };
    // The rule the engine applies, which the test switches between the
    // replies: none, or one of two. A rule makes a mask for every row, so
    // that no draft of the model's own is verified with it (the host cannot
    // mask a row for drafts it does not have yet); the n-gram drafter's are.
    let mode = Arc::new(AtomicUsize::new(NONE));
    let most = Arc::new(AtomicUsize::new(0));
    let rule: Rule = {
        let (mode, most) = (mode.clone(), most.clone());
        Arc::new(move |place, mask| match mode.load(Ordering::SeqCst) {
            // Which class of tokens may come at a place: the reply is made of
            // tokens whose id is the place modulo 3.
            BY_PLACE => {
                mask.fill(false);
                for token in (0..mask.tokens()).filter(|t| *t as usize % 3 == place % 3) {
                    mask.set(token, true);
                }
            }
            // The token the model says most often is not allowed. Drafts from
            // the text propose it all the same.
            WITHOUT_MOST => mask.set(most.load(Ordering::SeqCst) as u32, false),
            _ => {}
        })
    };
    let served = Served::start_drafting(&path, Some(rule));
    let plain = served.chat(json!({"draft": false})).await;
    let most_said = plain
        .ids
        .iter()
        .copied()
        .max_by_key(|id| plain.ids.iter().filter(|t| *t == id).count())
        .unwrap();
    most.store(most_said as usize, Ordering::SeqCst);

    let mut texts: Vec<Vec<Said>> = vec![Vec::new(), Vec::new()];
    for (name, which) in [("by place", BY_PLACE), ("without a token", WITHOUT_MOST)] {
        mode.store(which, Ordering::SeqCst);
        for (drafts, said) in texts.iter_mut().enumerate() {
            let before = served.drafts();
            for mut extra in [
                json!({"ignore_eos": true}),
                json!({"ignore_eos": true, "temperature": 0.8, "top_k": 40, "seed": 11}),
            ] {
                extra["draft"] = json!(drafts == 1);
                let reply = served.chat(extra.clone()).await;
                assert!(reply.ids.len() >= 30, "{name} {extra}: {:?}", reply.text);
                match which {
                    BY_PLACE => {
                        for (i, id) in reply.ids.iter().enumerate() {
                            assert_eq!(*id as usize % 3, i % 3, "{name} {extra}: token {i}");
                        }
                    }
                    _ => assert!(
                        !reply.ids.contains(&most_said),
                        "{name} {extra}: {:?}",
                        reply.pieces
                    ),
                }
                said.push(reply);
            }
            let (rows, accepted) = served.drafts();
            eprintln!(
                "RES mask {name}, drafts {}: {} draft rows verified, {} right",
                drafts == 1,
                rows - before.0,
                accepted - before.1
            );
        }
    }
    for (i, (a, b)) in texts[0].iter().zip(&texts[1]).enumerate() {
        eprintln!(
            "RES mask {i}: {} of {} tokens equal with drafts and without",
            a.ids.iter().zip(&b.ids).take_while(|(x, y)| x == y).count(),
            a.ids.len()
        );
    }
    served.runtime.shutdown();
}

// ---------------------------------------------------------------------------
// Drafts, and requests at the same time

/// Every setting of the sampler, with drafts and without (`"draft": false`):
/// the tokens and the probabilities are those without drafts. The sampler is
/// exact (its tests compare each record with its reference bit for bit, with
/// drafts and without them, on a model that has no error); the real model is
/// not: its logits for a row of a window differ a little from those of the row
/// alone, so a sampled text can part where two tokens are that close. A
/// request with mirostat has no drafts.
#[tokio::test]
async fn drafts_leave_the_tokens_of_every_setting_alone() {
    let Some(path) = drafting_model() else { return };
    let settings: Vec<(&str, Value)> = vec![
        ("greedy", json!({})),
        ("temperature", json!({"temperature": 0.8, "seed": 1})),
        (
            "top-k, top-p and min-p",
            json!({"temperature": 0.9, "top_k": 40, "top_p": 0.9, "min_p": 0.02, "seed": 2}),
        ),
        (
            "penalties",
            json!({"temperature": 0.7, "seed": 3, "repeat_penalty": 1.15,
            "frequency_penalty": 0.2, "presence_penalty": 0.1, "repeat_last_n": 64}),
        ),
        ("greedy with penalties", json!({"repeat_penalty": 1.3, "repeat_last_n": 32})),
        (
            "dry",
            json!({"temperature": 0.8, "seed": 4, "dry_multiplier": 0.8, "dry_penalty_last_n": 128}),
        ),
        (
            "bias",
            json!({"temperature": 0.8, "seed": 5, "logit_bias": [[" the", false], [" fox", 2.0]]}),
        ),
        (
            "typical, top-n-sigma, xtc, dynamic temperature",
            json!({"temperature": 0.8, "seed": 6,
            "typical_p": 0.9, "top_n_sigma": 3.0, "xtc_probability": 0.3, "dynatemp_range": 0.3}),
        ),
        (
            "an order of its own",
            json!({"temperature": 0.8, "seed": 7, "top_k": 60, "top_p": 0.9,
            "min_p": 0.02, "samplers": ["temperature", "min_p", "top_k", "top_p"]}),
        ),
        ("probabilities", json!({"temperature": 0.8, "seed": 8, "top_k": 40, "top_logprobs": 5})),
        (
            "probabilities after the chain",
            json!({"temperature": 0.8, "seed": 9, "top_k": 40,
            "top_p": 0.9, "n_probs": 5, "post_sampling_probs": true}),
        ),
        ("mirostat 2", json!({"temperature": 0.8, "seed": 10, "mirostat": 2})),
        ("mirostat 1", json!({"temperature": 0.8, "seed": 11, "mirostat": 1})),
        ("a stop string", json!({"temperature": 0.8, "seed": 12, "stop": ["lazy"]})),
    ];
    let served = Served::start_drafting(&path, None);
    let mut answers: Vec<Vec<Said>> = vec![Vec::new(), Vec::new()];
    for (drafts, said) in answers.iter_mut().enumerate() {
        for (name, extra) in &settings {
            let mut extra = extra.clone();
            extra["max_tokens"] = json!(100);
            extra["draft"] = json!(drafts == 1);
            let before = served.drafts();
            said.push(served.chat(extra).await);
            let rows = served.drafts().0 - before.0;
            if name.starts_with("mirostat") {
                assert_eq!(rows, 0, "{name}: a request with mirostat verified {rows} draft rows");
            }
        }
        let (rows, accepted) = served.drafts();
        eprintln!("RES drafts {}: {rows} draft rows verified, {accepted} right", drafts == 1);
    }
    let (rows, accepted) = served.drafts();
    assert!(rows > 0 && accepted > 0, "drafts were verified ({rows} rows, {accepted} right)");
    served.runtime.shutdown();
    let mut equal = 0;
    for ((name, _), (plain, drafted)) in settings.iter().zip(answers[0].iter().zip(&answers[1])) {
        let same = common(&plain.text, &drafted.text);
        let total = plain.text.chars().count();
        eprintln!("RES {name}: {same} of {total} characters equal with drafts and without");
        assert!(plain.ids.len() >= 8, "{name}: {:?}", plain.text);
        if plain.text == drafted.text && plain.ids == drafted.ids {
            equal += 1;
        } else {
            assert_ne!(*name, "greedy", "a greedy text parts");
            assert_ne!(*name, "greedy with penalties", "a greedy text parts");
        }
        assert!(same >= 100.min(total), "{name}: the texts part early, at character {same}");
        let together = plain.ids.iter().zip(&drafted.ids).take_while(|(a, b)| a == b).count();
        let worst = plain
            .scores
            .iter()
            .zip(&drafted.scores)
            .take(together)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 2e-2, "{name}: the probabilities move by {worst}");
    }
    assert!(equal * 4 >= settings.len() * 3, "only {equal} of {} texts are equal", settings.len());
}

/// A request has its own sampler row: what another request in the same steps
/// asks for does not reach it. One request alone and the same one next to
/// others that ask for penalties, a ban, probabilities or a different seed say
/// the same, with drafts and without.
#[tokio::test]
async fn requests_with_different_settings_at_once_do_not_affect_each_other() {
    let Some(path) = drafting_model() else { return };
    let a =
        json!({"temperature": 0.8, "top_k": 40, "seed": 21, "top_logprobs": 3, "max_tokens": 60});
    let b = json!({"temperature": 0.7, "seed": 22, "repeat_penalty": 1.4, "frequency_penalty": 0.5,
        "dry_multiplier": 1.0, "logit_bias": [[" the", false]], "max_tokens": 60});
    let c = json!({"temperature": 1.2, "seed": 23, "mirostat": 2, "ignore_eos": true,
        "n_probs": 4, "post_sampling_probs": true, "max_tokens": 60});
    let d = json!({"messages": [{"role": "user", "content": "Name three colours."}],
        "temperature": 0.5, "seed": 24, "typical_p": 0.9, "xtc_probability": 0.2, "max_tokens": 60});
    let served = Served::start_drafting(&path, None);
    for drafts in [false, true] {
        let with = |request: &Value| {
            let mut request = request.clone();
            request["draft"] = json!(drafts);
            request
        };
        let mut alone = Vec::new();
        for extra in [&a, &b, &c, &d] {
            alone.push(served.chat(with(extra)).await);
        }
        let together = tokio::join!(
            served.chat(with(&a)),
            served.chat(with(&b)),
            served.chat(with(&c)),
            served.chat(with(&d))
        );
        let together = [together.0, together.1, together.2, together.3];
        for (i, (alone, together)) in alone.iter().zip(&together).enumerate() {
            let same = alone.ids.iter().zip(&together.ids).take_while(|(x, y)| x == y).count();
            eprintln!(
                "RES drafts {drafts}, request {i}: {same} of {} tokens equal next to the others",
                alone.ids.len()
            );
            assert!(alone.ids.len() >= 4, "request {i}: {:?}", alone.text);
            // The model is not exact: the logits of a row depend a little on
            // the size of the step ("Rounding" in the architecture), so a
            // sampled text can part from the one it has alone where two tokens
            // are that close. What the others asked for must not show.
            let least = if drafts { 5 } else { 10 };
            assert!(same >= least.min(alone.ids.len()), "request {i} parts at token {same}");
            let worst = alone
                .scores
                .iter()
                .zip(&together.scores)
                .take(same)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-1, "request {i}: the probabilities move by {worst}");
        }
        // Each reply carries what its own request set and nothing else.
        let [a_said, b_said, c_said, d_said] = &together;
        assert!(a_said.alternatives.iter().all(|a| a.len() == 3), "A reports 3 alternatives");
        assert!(a_said.pieces.contains(&" the".to_owned()), "A has no ban: {:?}", a_said.pieces);
        assert!(b_said.alternatives.iter().all(Vec::is_empty), "B reports no alternatives");
        assert!(!b_said.pieces.contains(&" the".to_owned()), "B bans ' the': {:?}", b_said.pieces);
        assert!(c_said.alternatives.iter().all(|a| !a.is_empty() && a.len() <= 4));
        assert!(c_said.scores.iter().all(|p| (0.0..=1.0).contains(p)), "C reports probabilities");
        assert_eq!(c_said.ids.len(), 60, "C ignores the end: {:?}", c_said.text);
        assert!(d_said.alternatives.iter().all(Vec::is_empty), "D reports no alternatives");
    }
    served.runtime.shutdown();
}

/// `n` completions are requests of their own with seeds one after the other,
/// each with its own probabilities, and the usage counts the prompt once.
#[tokio::test]
async fn n_completions_are_requests_of_their_own_with_their_own_probabilities() {
    let Some(path) = model_path() else { return };
    let served = Served::start(&path, None);
    // A text that the model is not sure of, so that the seeds make a difference.
    let story = json!([{"role": "user", "content": "Tell me a very short story about a lighthouse keeper."}]);
    let sampled = json!({
        "messages": story, "temperature": 1.0, "seed": 50, "top_logprobs": 2, "max_tokens": 30,
    });
    let mut body = chat_body(sampled.clone());
    body["n"] = json!(3);
    let value = served.answer(CHAT, &body).await;
    let choices = value["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3);
    let mut generated = 0;
    let mut replies = Vec::new();
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], i);
        let said = said(choice);
        generated += said.ids.len() as u64;
        let mut one = sampled.clone();
        one["seed"] = json!(50 + i as u64);
        let alone = served.chat(one).await;
        // The reply is the one the seed gives, up to the model's rounding: the
        // choices ran side by side and the first of them computed the prompt,
        // so a sampled text can part from the one it has alone where two
        // tokens are that close (it did at the fifth token of 24 in a run of
        // the binary). The server tests with the mock engine check the seeds.
        let same = said.ids.iter().zip(&alone.ids).take_while(|(a, b)| a == b).count();
        eprintln!("RES n choice {i}: {same} of {} tokens equal the request alone", said.ids.len());
        assert!(said.ids.len() >= 20, "choice {i}: {:?}", said.text);
        let worst = said
            .scores
            .iter()
            .zip(&alone.scores)
            .take(same)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-1, "choice {i}: the probabilities move by {worst}");
        replies.push(said.ids);
    }
    assert!(
        replies[0] != replies[1] || replies[1] != replies[2],
        "choices with different seeds say different things"
    );
    // One prompt, and the tokens of all the replies (a token that ends one
    // is not among the probabilities).
    let counted = value["usage"]["completion_tokens"].as_u64().unwrap();
    assert!((generated..=generated + 3).contains(&counted), "{counted} against {generated}");
    served.runtime.shutdown();
}
