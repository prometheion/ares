//! A conversation whose client sends the reasoning of every reply back
//! continues from the prefix cache with nothing computed again, through the
//! server, with the real Qwen3.5-0.8B on a GPU.
//!
//! The server runs in this process, over the real engine, so that the test can
//! read the engine's report of each request (`Snapshot::requests`) as well as
//! `/metrics`. Requests go through the router without a socket.
//!
//! The 0.8B's own chat template writes the thinking of a turn into the prompt
//! only after the last user message, so a conversation with it cannot continue
//! from its own replies. The model here is served with a template that has the
//! structure of the Qwen3.8 templates instead: the text of the template of
//! Qwen3.8-27B if `ARES_TEST_QWEN38_27B` names that model file (the template is
//! read from the file, not copied), a short one written for this test if not.
//! A test with the 0.8B's own template shows what that costs.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL` (the 0.8B);
//! `ARES_TEST_DEVICE` names the device (default 0). Lines that start with
//! `RES` are measurements.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test --release -p ares-server --test reasoning_cache -- --test-threads=1
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ares_api::{ChatFormat, EngineError, ModelCard, Prompt, Role};
use ares_cuda::{Device, Precision, Wait};
use ares_gguf::Gguf;
use ares_model::chat;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, GpuConfig, GpuExecutor, LinkSpec, Model, ModelError,
    PrefixCache, RequestReport, Runtime, RuntimeConfig, Split,
};
use ares_sampler::SampleParams;
use ares_state::PageLayout;
use ares_tokenizer::{
    ChatOptions, ChatTemplate, EncodeOptions, Json, Message, ReasoningFormat, Tokenizer,
};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, header};
use serde_json::{Value, json};
use tower::ServiceExt;

const PAGE: u32 = 128;

/// The structure of the templates of the Qwen3.8 models, shortened: the thinking
/// of an assistant turn between think tags, trimmed, kept for earlier turns
/// unless `preserve_thinking` is off, an end of the prompt that opens the block
/// or closes an empty one.
const QWEN_LIKE: &str = r#"
{%- set ns = namespace(last_query_index=0) %}
{%- for message in messages %}
    {%- if message.role == "user" %}{%- set ns.last_query_index = loop.index0 %}{%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- set content = message.content|trim %}
    {%- if message.role == "user" or message.role == "system" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning = (message.reasoning_content if message.reasoning_content is string else '')|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>assistant\n<think>\n' + reasoning + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>assistant\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"#;

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

/// The template to serve the model with: read from the 27B's file if there is
/// one, the shortened one otherwise.
fn preserving_template() -> (String, &'static str) {
    if let Some(path) = std::env::var_os("ARES_TEST_QWEN38_27B") {
        let gguf = Gguf::open(path).unwrap();
        let source = gguf.str("tokenizer.chat_template").expect("a template").to_owned();
        return (source, "the template of Qwen3.8-27B");
    }
    (QWEN_LIKE.to_owned(), "a template with the structure of Qwen3.8's")
}

/// What the engine was given and what it generated, in order: every prompt of a
/// chat and every token after it, so that a test can say where a prompt parts
/// from the sequence that came before it.
#[derive(Default)]
struct Trace(Mutex<Vec<Traced>>);

enum Traced {
    Prompt(Vec<u32>),
    Token(u32),
}

impl Trace {
    /// Where the prompt that follows the request `turn` (counting from 0) parts
    /// from the prompt and the tokens of that request.
    fn parting(&self, turn: usize, tokenizer: &Tokenizer) -> String {
        let events = self.0.lock().unwrap();
        let (mut prompts, mut generated): (Vec<&Vec<u32>>, Vec<Vec<u32>>) =
            (Vec::new(), Vec::new());
        for event in events.iter() {
            match event {
                Traced::Prompt(prompt) => {
                    prompts.push(prompt);
                    generated.push(Vec::new());
                }
                Traced::Token(token) => generated.last_mut().expect("a prompt first").push(*token),
            }
        }
        let mut sequence = prompts[turn].clone();
        sequence.extend(&generated[turn]);
        let next = prompts[turn + 1];
        let kept = sequence.iter().zip(next).take_while(|(a, b)| a == b).count();
        let words = |ids: &[u32]| {
            let ids = &ids[kept.saturating_sub(4)..ids.len().min(kept + 4)];
            ids.iter()
                .map(|id| String::from_utf8_lossy(tokenizer.piece(*id, true).unwrap()).into_owned())
                .collect::<Vec<_>>()
        };
        format!(
            "the prompt after request {} repeats {kept} of the {} tokens before it ({} of them \
             the prompt); there the sequence goes on {:?} and the prompt {:?}",
            turn + 1,
            sequence.len(),
            prompts[turn].len(),
            words(&sequence),
            words(next)
        )
    }
}

/// The 0.8B with another chat template: everything is the model's own but the
/// prompt of a conversation and what the template declares.
struct Templated {
    trace: Arc<Trace>,
    inner: Arc<Qwen35>,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    /// What the server was started with: the thinking of earlier turns stays.
    chat: ChatOptions,
    /// What the template declares.
    format: ChatFormat,
}

impl Model for Templated {
    fn card(&self) -> ModelCard {
        self.inner.card()
    }
    fn vocab(&self) -> u32 {
        self.inner.vocab()
    }
    fn ends(&self, token: u32) -> bool {
        self.inner.ends(token)
    }
    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        let Prompt::Chat { messages, variables, .. } = prompt else {
            return self.inner.encode(prompt);
        };
        let messages: Vec<Message> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let mut message = Message::new(role, m.content.clone());
                message.reasoning_content = m.reasoning_content.clone().unwrap_or_default();
                message
            })
            .collect();
        let options = chat::with_variables(&self.chat, variables);
        let text = self
            .template
            .render(&messages, &[], &options)
            .map_err(|e| EngineError::InvalidRequest(format!("chat template: {e}")))?;
        let tokens =
            self.tokenizer.encode(&text, EncodeOptions { add_special: true, parse_special: true });
        self.trace.0.lock().unwrap().push(Traced::Prompt(tokens.clone()));
        Ok(tokens)
    }
    fn chat_format(&self) -> Option<ChatFormat> {
        Some(self.format.clone())
    }
    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        self.trace.0.lock().unwrap().push(Traced::Token(token));
        self.inner.piece(token, text);
    }
    fn context(&self) -> ContextSpec {
        self.inner.context()
    }
    fn split(&self) -> Split {
        self.inner.split()
    }
    fn devices(&self) -> usize {
        self.inner.devices()
    }
    fn links(&self) -> LinkSpec {
        self.inner.links()
    }
    fn max_rows(&self) -> u32 {
        self.inner.max_rows()
    }
    fn logit_rows(&self) -> u32 {
        self.inner.logit_rows()
    }
    fn draft_rows(&self) -> u32 {
        self.inner.draft_rows()
    }
    fn sampling(&self) -> SampleParams {
        self.inner.sampling()
    }
    fn attention_span(&self) -> u32 {
        self.inner.attention_span()
    }
    fn own_drafts(&self) -> u32 {
        self.inner.own_drafts()
    }
    fn prefix_cache(&self) -> Option<PrefixCache> {
        self.inner.prefix_cache()
    }
    fn page_entry(&self, page: u32) -> u32 {
        self.inner.page_entry(page)
    }
    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        self.inner.page_layout(cells_per_page)
    }
    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        self.inner.open(cx)
    }
}

/// The server over the engine on the 0.8B.
struct Server {
    runtime: Arc<Runtime>,
    app: Router,
    /// What the engine saw, when the model was served with a template of its own.
    trace: Arc<Trace>,
    path: PathBuf,
    /// How many requests the engine has reported.
    reported: usize,
}

/// What a request came to.
#[derive(Debug)]
struct Answer {
    content: String,
    reasoning: String,
    finish: String,
    usage: Value,
    report: RequestReport,
}

impl Answer {
    fn prompt(&self) -> u64 {
        self.usage["prompt_tokens"].as_u64().unwrap()
    }

    fn completion(&self) -> u64 {
        self.usage["completion_tokens"].as_u64().unwrap()
    }

    fn cached(&self) -> u64 {
        self.usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap()
    }
}

impl Server {
    /// Starts the engine on the 0.8B. `template` is the chat template to serve
    /// it with, or `None` for the model's own.
    fn start(path: &PathBuf, template: Option<&str>) -> Self {
        let options = Options {
            precision: Precision::Exact,
            max_rows: 128,
            logit_rows: 16,
            window_rows: 4,
            pipeline_rows: 32,
            ..Options::default()
        };
        // What the server starts with: the thinking of earlier turns stays.
        let chat = ChatOptions {
            variables: vec![("preserve_reasoning".to_owned(), Json::Bool(true))],
            ..ChatOptions::default()
        };
        let qwen = Arc::new(
            Qwen35::new(Gguf::open(path).unwrap(), "q", &Spread::Single, options, 8192)
                .unwrap()
                .with_chat_options(chat.clone()),
        );
        let checkpoint = qwen.prefix_cache().expect("qwen35 takes part").bytes();
        let config = RuntimeConfig {
            pages: 64,
            cells_per_page: PAGE,
            max_slots: 2,
            window_rows: 4,
            checkpoint_prompt_interval: PAGE,
            checkpoint_generation_interval: PAGE,
            checkpoint_budget: 16 * checkpoint,
            ..RuntimeConfig::default()
        };
        let trace = Arc::new(Trace::default());
        let model: Arc<dyn Model> = match template {
            None => qwen,
            Some(source) => {
                let template = ChatTemplate::new(source, "", "").unwrap();
                let thinking = ReasoningFormat::detect(&template)
                    .map(|format| chat::thinking_format(format, &chat));
                Arc::new(Templated {
                    trace: trace.clone(),
                    tokenizer: Tokenizer::from_gguf(&Gguf::open(path).unwrap()).unwrap(),
                    template,
                    inner: qwen,
                    chat,
                    format: ChatFormat { thinking, tools: None },
                })
            }
        };
        let gpu = GpuConfig::new(&config, &*model);
        let ordinal =
            std::env::var("ARES_TEST_DEVICE").ok().and_then(|d| d.parse().ok()).unwrap_or(0);
        let device = Device::open(ordinal, Wait::Block).unwrap();
        let executor = GpuExecutor::start(model.clone(), vec![device], gpu).unwrap();
        let runtime = Arc::new(Runtime::start(model, Box::new(executor), config).unwrap());
        let app = ares_server::router(runtime.clone());
        let reported = runtime.snapshot().expect("the engine runs").requests.len();
        Self { runtime, app, trace, path: path.clone(), reported }
    }

    async fn post(&self, body: &Value) -> (u16, String) {
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = self.app.clone().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    /// Where the prompt after the request `turn` parts from what came before it.
    fn parting(&self, turn: usize) -> String {
        let tokenizer = Tokenizer::from_gguf(&Gguf::open(&self.path).unwrap()).unwrap();
        self.trace.parting(turn, &tokenizer)
    }

    /// One metric of `/metrics`.
    async fn metric(&self, name: &str) -> u64 {
        let response = self
            .app
            .clone()
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix(&format!("ares_{name} ")))
            .unwrap_or_else(|| panic!("no metric {name}"))
            .parse()
            .unwrap()
    }

    /// Asks `messages`, plain or streamed, and returns the answer with the
    /// engine's report of the request.
    async fn ask(&mut self, messages: &[Value], stream: bool, limit: u32) -> Answer {
        let mut body = json!({
            "model": "q", "messages": messages, "max_tokens": limit, "temperature": 0,
        });
        if stream {
            body["stream"] = json!(true);
            body["stream_options"] = json!({"include_usage": true});
        }
        let (status, text) = self.post(&body).await;
        assert_eq!(status, 200, "{text}");
        let (content, reasoning, finish, usage) = if stream {
            let chunks: Vec<Value> = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).unwrap())
                .collect();
            let delta = |field: &str| -> String {
                chunks
                    .iter()
                    .filter_map(|chunk| chunk["choices"].get(0)?["delta"][field].as_str())
                    .collect()
            };
            let finish = chunks
                .iter()
                .find_map(|chunk| chunk["choices"].get(0)?["finish_reason"].as_str())
                .unwrap()
                .to_owned();
            let usage = chunks.last().unwrap()["usage"].clone();
            (delta("content"), delta("reasoning_content"), finish, usage)
        } else {
            let body: Value = serde_json::from_str(&text).unwrap();
            let choice = &body["choices"][0];
            (
                choice["message"]["content"].as_str().unwrap().to_owned(),
                choice["message"]["reasoning_content"].as_str().unwrap_or_default().to_owned(),
                choice["finish_reason"].as_str().unwrap().to_owned(),
                body["usage"].clone(),
            )
        };
        // The engine reports a request when it has ended it.
        let begun = Instant::now();
        let report = loop {
            let reports = self.runtime.snapshot().expect("the engine runs").requests;
            if reports.len() > self.reported {
                self.reported = reports.len();
                break *reports.last().unwrap();
            }
            assert!(begun.elapsed() < Duration::from_secs(20), "the engine never reported");
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        Answer { content, reasoning, finish, usage, report }
    }
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

/// The reply as the client sends it back: the reasoning too, or the content only.
fn sent_back(answer: &Answer, with_reasoning: bool) -> Value {
    let mut message = json!({"role": "assistant", "content": answer.content});
    if with_reasoning {
        message["reasoning_content"] = json!(answer.reasoning);
    }
    message
}

/// A system message of about 300 tokens, so that the history is several pages
/// of the pool (128 cells each) before the first reply.
fn system() -> String {
    let rules: Vec<String> = (1..=14)
        .map(|n| {
            format!(
                "Rule {n}: when a sum has {} terms the answer is a whole number and is written in digits.",
                n + 1
            )
        })
        .collect();
    format!("You are a calculator. {}", rules.join(" "))
}

const QUESTIONS: [&str; 3] = [
    "What is 2 + 2? Answer with the number only.",
    "And what is 3 + 3? Answer with the number only.",
    "And what is 5 + 5? Answer with the number only.",
];

/// What a conversation is like.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Short questions: the model thinks for a few hundred tokens and stops by
    /// itself, and the history grows by the replies.
    Short,
    /// A system message of three pages first: the model thinks for long, usually
    /// until the limit of tokens cuts it off, and the prompt parts from a reply that
    /// is not repeated beyond a page boundary.
    Long,
}

impl Kind {
    /// The tokens a reply may have. The model is asked greedily: a sampled reply
    /// has other tokens than the tokenizer would make of its text once in a while,
    /// and the next prompt parts there (see `reasoning/templates.rs` in the server).
    fn limit(self) -> u32 {
        match self {
            Kind::Short => 1500,
            Kind::Long => 600,
        }
    }
}

/// Plays three turns, the client keeping what it was given, and returns the
/// answers.
async fn conversation(server: &mut Server, kind: Kind, with_reasoning: bool) -> Vec<Answer> {
    let mut messages = match kind {
        Kind::Short => vec![user(QUESTIONS[0])],
        Kind::Long => vec![json!({"role": "system", "content": system()}), user(QUESTIONS[0])],
    };
    let mut answers = Vec::new();
    for (turn, question) in QUESTIONS.iter().enumerate() {
        if turn > 0 {
            messages.push(user(question));
        }
        // The first turn is streamed, the others are not: the same reply either way.
        let answer = server.ask(&messages, turn == 0, kind.limit()).await;
        eprintln!(
            "RES turn {}: {} prompt tokens, {} cached, {} computed, {} replayed, {} generated \
             ({} chars of reasoning, {} of content, {})",
            turn + 1,
            answer.prompt(),
            answer.cached(),
            answer.report.computed_rows,
            answer.report.replayed_rows,
            answer.completion(),
            answer.reasoning.chars().count(),
            answer.content.chars().count(),
            answer.finish,
        );
        messages.push(sent_back(&answer, with_reasoning));
        answers.push(answer);
    }
    answers
}

#[tokio::test]
async fn a_conversation_whose_client_sends_the_reasoning_back_computes_only_the_new_message() {
    let Some(path) = model_path() else { return };
    let (template, which) = preserving_template();
    eprintln!("RES served with {which}");
    let mut server = Server::start(&path, Some(&template));
    let mut stopped = 0;
    let mut turns = 0;
    for kind in [Kind::Short, Kind::Long] {
        let first = server.reported;
        let answers = conversation(&mut server, kind, true).await;
        for answer in &answers {
            assert!(!answer.reasoning.is_empty(), "the model thought: {answer:?}");
            assert!(!answer.content.contains("</think>") && !answer.reasoning.contains("</think>"));
            // The engine reported what the answer's usage says.
            assert_eq!(u64::from(answer.report.prompt_tokens), answer.prompt());
            assert_eq!(u64::from(answer.report.cached_tokens), answer.cached());
            stopped += usize::from(answer.finish == "stop");
            turns += 1;
        }
        // Turns two and three start where the reply of the turn before ended,
        // whether the model ended it or the limit of tokens did: the whole sequence of
        // that turn is cached but the last token, which is generated and never
        // computed; nothing is replayed; and what is computed is the new message and
        // what the template writes around it.
        for turn in 1..3 {
            let (before, now) = (&answers[turn - 1], &answers[turn]);
            let sequence = before.prompt() + before.completion();
            let at = turn + 1;
            if now.cached() != sequence - 1 {
                panic!(
                    "turn {at}: {} tokens are cached, not the {} of the turn before: {}",
                    now.cached(),
                    sequence - 1,
                    server.parting(first + turn - 1)
                );
            }
            assert_eq!(now.report.replayed_rows, 0, "turn {at}: nothing is replayed");
            assert_eq!(
                u64::from(now.report.computed_rows),
                now.prompt() - now.cached(),
                "turn {at}: only the rows that are not cached are computed"
            );
            // And those are few: the last token of the reply, the line break after it,
            // the new message and the start of the next reply.
            assert!(now.report.computed_rows < 40, "turn {at}: {:?}", now.report);
        }
    }
    assert!(stopped >= 1, "{stopped} of {turns} replies ended with the end token");
    // The counters say the same: the cache was used four times, and nothing was
    // replayed in the whole run.
    assert_eq!(server.metric("replayed_rows_total").await, 0);
    assert!(server.metric("prefix_hits_total").await >= 4);
    assert_eq!(server.metric("requests_finished_total").await, 6);
}

#[tokio::test]
async fn a_client_that_drops_the_reasoning_pays_for_it_in_every_turn() {
    let Some(path) = model_path() else { return };
    let (template, which) = preserving_template();
    eprintln!("RES served with {which}");
    let mut server = Server::start(&path, Some(&template));
    // The same conversation, the client sending back only what `content` said.
    let answers = conversation(&mut server, Kind::Long, false).await;

    for turn in 1..3 {
        let (before, now) = (&answers[turn - 1], &answers[turn]);
        // The template writes an empty block where the prompt of the turn before had
        // "<think>\n", so the new prompt parts from that sequence at the end of the
        // old prompt: its last token, the newline, is one token with the next. A
        // page is cached whole or not at all, so what is cached is the whole pages
        // before that, and the rest of the old prompt is computed again with the new
        // message. (The reply the model generated is not in the new prompt, so it is
        // not computed again; it is gone.)
        let parts = before.prompt() - 1;
        assert_eq!(now.cached(), parts / u64::from(PAGE) * u64::from(PAGE), "turn {}", turn + 1);
        assert_eq!(u64::from(now.report.computed_rows), now.prompt() - now.cached());
        assert!(u64::from(now.report.computed_rows) >= parts % u64::from(PAGE));
    }
}

#[tokio::test]
async fn with_its_own_template_the_0_8b_cannot_continue_from_its_replies_whatever_is_sent_back() {
    let Some(path) = model_path() else { return };
    let mut server = Server::start(&path, None);
    // The template writes the thinking of a turn into the prompt only after the
    // last user message: in the next request the reply is an earlier turn.
    let answers = conversation(&mut server, Kind::Long, true).await;
    for turn in 1..3 {
        let (before, now) = (&answers[turn - 1], &answers[turn]);
        // The prompt of the turn before ended with "assistant\n<think>\n"; this one
        // has the reply without its thinking after "assistant\n". Whole pages only.
        let parts = before.prompt() - 2;
        assert_eq!(now.cached(), parts / u64::from(PAGE) * u64::from(PAGE), "turn {}", turn + 1);
        assert_eq!(u64::from(now.report.computed_rows), now.prompt() - now.cached());
    }
}
