//! `qwen4exp` as the runtime drives it: a made-up model of the architecture
//! on the GPUs, through the device threads, the step record, the page tables
//! of the conductor and the expert store, against the model's own session.
//!
//! The tests open GPUs and are skipped unless `ARES_TEST_CUDA=1` is set; the
//! placements over two devices need two. Every placement is run: the layers
//! on one device, a range of layers on each of two, and both devices on every
//! layer. The model's vocabulary is a placeholder, so the tests give prompts
//! as the text `t5 t9 ...` and read the answers back the same way.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use ares_api::{
    BiasKey, ChatFormat, Engine, EngineError, Event, EventSink, GenerateRequest, LogitBias,
    LogprobRequest, ModelCard, Prompt, RequestId, Sampling, TokenProbs,
};
use ares_core::GgmlType;
use ares_cuda::{CacheFormat, Device, Wait, device_count};
use ares_gguf::{Gguf, GgufBuilder};
use ares_model::qwen4exp::device::{
    Options, Qwen4Exp, Rows, Session, SessionOptions, Spread, Wanted,
};
use ares_model::qwen4exp::synthetic::{NgramShape, Shape, Synthetic, Types};
use ares_model::qwen4exp::{Config, LogitRows, Reference, SequenceState, Weights};
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, GpuConfig, LinkSpec, Model, ModelError, PrefixCache,
    Runtime, RuntimeConfig, Split,
};
use ares_state::PageLayout;
use ares_tokenizer::{ChatOptions, Json, Tokenizer};

const MAX_ROWS: u32 = 8;
/// The rows of a step that reads a slice of a prompt on the many-row kernels.
const WIDE_ROWS: u32 = 64;
const CELLS: u32 = 512;
const SLOTS: u32 = 2;

fn enabled() -> bool {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return false;
    }
    true
}

/// A made-up model with every part of the architecture at sizes the kernels
/// take.
fn model(seed: u64) -> Synthetic {
    let shape = Shape {
        n_embd: 256,
        mixer_rank: 32,
        n_head: 8,
        n_head_kv: 2,
        head_dim: 128,
        rope_dim: 64,
        rope_sections: [11, 11, 10, 0],
        indexer_heads: 3,
        indexer_dim: 128,
        n_k_heads: 2,
        n_v_heads: 4,
        state_size: 128,
        n_expert: 48,
        n_expert_used: 3,
        expert_ff: 64,
        shared_ff: 64,
        ngram: Some(NgramShape {
            layer: 1,
            ngram_size: 3,
            heads_per_ngram: 4,
            conv_kernel: 4,
            eos_token: 7,
            image_token: 9,
            head_vocab: vec![101, 103, 107, 109, 113, 127, 131, 137],
            spare_rows: 4,
        }),
        ..Shape::small()
    };
    Synthetic { shape, seed, types: Types::Shipped, lazy_type: Some(GgmlType::IQ4_NL) }
}

fn gguf_path(seed: u64) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4rt-{seed}.gguf"));
    std::fs::write(&path, model(seed).bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0))
        .expect("the test directory is writable");
    path
}

/// How the model is spread over the devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// Every layer on the one device.
    One,
    /// A range of layers on each of two devices.
    Layers,
    /// Both devices on every layer.
    Shared,
}

fn placements() -> Vec<Placement> {
    let mut all = vec![Placement::One];
    if device_count().expect("the CUDA driver") >= 2 {
        all.extend([Placement::Layers, Placement::Shared]);
    }
    all
}

fn open_devices(placement: Placement) -> Vec<Device> {
    let count = if placement == Placement::One { 1 } else { 2 };
    (0..count).map(|d| Device::open(d, Wait::Spin).expect("open device")).collect()
}

fn options(
    placement: Placement,
    n_layer: usize,
    pages: u32,
    window_rows: u32,
    rows: u32,
) -> SessionOptions {
    SessionOptions {
        part: Options {
            kv: (CacheFormat::Q8, CacheFormat::Q8),
            max_rows: rows,
            logit_rows: MAX_ROWS,
            window_rows,
            max_slots: SLOTS,
            cells: CELLS,
            pool_pages: pages,
            ..Options::default()
        },
        // The tests share the cards with whatever else runs.
        headroom: 512 << 20,
        spread: match placement {
            Placement::Shared => Spread::Tensor {
                mixer: Some(0),
                core_share: Some(0.5),
                shared_on_mixer: Some(false),
            },
            _ => Spread::Layers,
        },
        bounds: (placement == Placement::Layers).then(|| vec![0, n_layer / 2, n_layer]),
        // Every expert has a slot: none is to be evicted between steps.
        reserve: Some(0),
        cpu_workers: Some(2),
        ..SessionOptions::default()
    }
}

/// A tokenizer of the 256 bytes alone, which the model needs to exist and the
/// tests do not use: their prompts are token ids. It has the chat template
/// `template`, if any.
fn byte_tokenizer(template: Option<&str>) -> Tokenizer {
    // The characters of GPT-2's byte-level alphabet.
    let tokens: Vec<String> = (0..=255u8)
        .map(|byte| {
            let printable = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
            let moved =
                (0..byte).filter(|b| !matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)).count();
            let at = if printable { u32::from(byte) } else { 0x100 + moved as u32 };
            char::from_u32(at).unwrap().to_string()
        })
        .collect();
    let mut file = GgufBuilder::new()
        .str("general.architecture", "qwen4exp")
        .str("tokenizer.ggml.model", "gpt2")
        .str("tokenizer.ggml.pre", "qwen35")
        .str_array("tokenizer.ggml.tokens", &tokens)
        .i32_array("tokenizer.ggml.token_type", &vec![1; 256])
        .str_array::<&str>("tokenizer.ggml.merges", &[]);
    if let Some(template) = template {
        file = file.str("tokenizer.chat_template", template);
    }
    let bytes = file.build();
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(if template.is_some() {
        "q4rt-tokenizer-chat.gguf"
    } else {
        "q4rt-tokenizer.gguf"
    });
    std::fs::write(&path, bytes).unwrap();
    Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).expect("a tokenizer of bytes")
}

/// The model of the runtime with the vocabulary of the made-up file: a
/// prompt is `t5 t9 ...`, a token prints as `t5 `.
struct Made(Arc<Qwen4Exp>);

impl Model for Made {
    fn card(&self) -> ModelCard {
        self.0.card()
    }
    fn vocab(&self) -> u32 {
        self.0.vocab()
    }
    fn ends(&self, _token: u32) -> bool {
        false
    }
    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        let Prompt::Text(text) = prompt else {
            return Err(EngineError::Unsupported("the made-up model has no chat template".into()));
        };
        text.split_whitespace()
            .map(|word| {
                word.strip_prefix('t')
                    .and_then(|n| n.parse().ok())
                    .ok_or_else(|| EngineError::InvalidRequest(format!("{word:?} is no token")))
            })
            .collect()
    }
    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        text.extend_from_slice(format!("t{token} ").as_bytes());
    }
    fn context(&self) -> ContextSpec {
        self.0.context()
    }
    fn split(&self) -> Split {
        self.0.split()
    }
    fn devices(&self) -> usize {
        self.0.devices()
    }
    fn links(&self) -> LinkSpec {
        self.0.links()
    }
    fn pinned_ram(&self) -> u64 {
        self.0.pinned_ram()
    }
    fn max_rows(&self) -> u32 {
        self.0.max_rows()
    }
    fn logit_rows(&self) -> u32 {
        self.0.logit_rows()
    }
    fn draft_rows(&self) -> u32 {
        self.0.draft_rows()
    }
    fn attention_span(&self) -> u32 {
        self.0.attention_span()
    }
    fn prefix_cache(&self) -> Option<PrefixCache> {
        self.0.prefix_cache()
    }
    fn page_entry(&self, page: u32) -> u32 {
        self.0.page_entry(page)
    }
    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        self.0.page_layout(cells_per_page)
    }
    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        self.0.open(cx)
    }
}

/// The runtime over the made-up model on `placement`, with `pages` pages of
/// 128 cells in the pool.
fn start(
    seed: u64,
    placement: Placement,
    pages: u32,
    config: RuntimeConfig,
) -> (Runtime, Arc<Qwen4Exp>) {
    start_rows(seed, placement, pages, config, MAX_ROWS)
}

/// [`start`] with steps of up to `rows` rows.
fn start_rows(
    seed: u64,
    placement: Placement,
    pages: u32,
    config: RuntimeConfig,
    rows: u32,
) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let opts = options(placement, n_layer, pages, MAX_ROWS, rows);
    let inner =
        Arc::new(Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap());
    let config = RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..config };
    let model: Arc<dyn Model> = Arc::new(Made(inner.clone()));
    let mut gpu = GpuConfig::new(&config, &*model);
    // Room for the tier of experts next to what the executor needs.
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// The tokens of the text `t5 t9 ...` a request wrote.
fn tokens_of(text: &str) -> Vec<u32> {
    text.split_whitespace().map(|w| w[1..].parse().unwrap()).collect()
}

fn ask(runtime: &Runtime, id: u64, prompt: &[u32], max_tokens: u32) -> Vec<u32> {
    let rx = submit(runtime, id, prompt, max_tokens);
    rx.recv_timeout(Duration::from_secs(300)).expect("an answer in time").unwrap()
}

/// The tokens a request with the sampler settings `sampling` wrote, with the
/// probabilities it reported, and whether drafts were allowed for it.
fn ask_sampled(
    runtime: &Runtime,
    id: u64,
    prompt: &[u32],
    max_tokens: u32,
    sampling: Sampling,
    draft: bool,
) -> (Vec<u32>, Vec<TokenProbs>) {
    let (tx, rx) = mpsc::channel();
    let said = Arc::new(Mutex::new((String::new(), Vec::new())));
    let sink = {
        let said = said.clone();
        EventSink::new(move |event| {
            match event {
                Event::Text(piece) => said.lock().unwrap().0.push_str(&piece),
                Event::Logprobs(probs) => said.lock().unwrap().1.extend(probs),
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
    let words: Vec<String> = prompt.iter().map(|t| format!("t{t}")).collect();
    runtime
        .submit(
            GenerateRequest {
                model: "made".into(),
                prompt: Prompt::Text(words.join(" ")),
                sampling,
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft,
                structure: None,
                id: RequestId(id),
            },
            sink,
        )
        .unwrap();
    rx.recv_timeout(Duration::from_secs(300)).expect("an answer in time").unwrap();
    let (text, probs) = std::mem::take(&mut *said.lock().unwrap());
    (tokens_of(&text), probs)
}

/// Puts every expert into a VRAM slot.
///
/// An expert that is in no slot is computed on the CPU or uploaded, whichever
/// the measured costs say at that moment, and the CPU's arithmetic is not the
/// device's: a made-up model has near ties that the difference flips. What a
/// test compares with the session's answer, whose experts all have slots, is
/// therefore an answer from a store that was filled.
fn warm(model: &Qwen4Exp) {
    let n_experts = model.config().experts.count as u32;
    model
        .fill(&mut |_| (0..n_experts).collect())
        .expect("the parts are open")
        .expect("the tiers fill");
    let (vram_slots, _, _) = model.expert_slots().expect("the parts are open");
    assert!(
        vram_slots >= model.config().n_layer * model.config().experts.count,
        "every expert has a slot"
    );
    model.take_store_stats();
}

/// No expert a step picked since `warm` was missing from its slot.
fn assert_nothing_missed(model: &Qwen4Exp, what: &str) {
    let stats = model.take_store_stats().expect("the parts are open");
    assert!(stats.picks > 0 && stats.picks == stats.hits, "{what}: {stats:?}");
}

fn submit(
    runtime: &Runtime,
    id: u64,
    prompt: &[u32],
    max_tokens: u32,
) -> mpsc::Receiver<Result<Vec<u32>, String>> {
    let (tx, rx) = mpsc::channel();
    let text = Arc::new(Mutex::new(String::new()));
    let sink = {
        let text = text.clone();
        EventSink::new(move |event| {
            match event {
                Event::Text(piece) => text.lock().unwrap().push_str(&piece),
                Event::Done { .. } => {
                    let _ = tx.send(Ok(tokens_of(&text.lock().unwrap())));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
                Event::Logprobs(_) => {}
            }
            true
        })
    };
    let words: Vec<String> = prompt.iter().map(|t| format!("t{t}")).collect();
    runtime
        .submit(
            GenerateRequest {
                model: "made".into(),
                prompt: Prompt::Text(words.join(" ")),
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
    rx
}

/// The greedy tokens the session writes after `prompt`.
fn session_answer(seed: u64, placement: Placement, prompt: &[u32], n: usize) -> Vec<u32> {
    let devices = open_devices(placement);
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let mut opts = options(placement, c.n_layer, 0, MAX_ROWS, MAX_ROWS);
    opts.part.pool_pages = 0;
    opts.ram_tier = Some(64 << 20);
    let mut session = Session::open(&gguf, &c, &w, devices, opts).expect("the session opens");
    let n_expert = c.experts.count as u32;
    session.fill(&mut |_| (0..n_expert).collect()).expect("the tiers fill");
    let chunks = prompt.chunks(MAX_ROWS as usize).count();
    let mut pick = 0;
    for (i, chunk) in prompt.chunks(MAX_ROWS as usize).enumerate() {
        let wanted = if i + 1 == chunks { Wanted::Last } else { Wanted::None };
        let picks =
            session.step(&[Rows { slot: 0, tokens: chunk, direct: true, wanted }]).expect("a step");
        pick = picks.first().copied().unwrap_or(pick);
    }
    let mut answer = vec![pick];
    while answer.len() < n {
        let token = [answer[answer.len() - 1]];
        let picks = session
            .step(&[Rows { slot: 0, tokens: &token, direct: true, wanted: Wanted::Last }])
            .expect("a step");
        answer.push(picks[0]);
    }
    answer
}

/// The greedy tokens the CPU reference writes after `prompt`.
fn reference_answer(seed: u64, prompt: &[u32], n: usize) -> Vec<u32> {
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut state = SequenceState::new(&c);
    let arg_max = |row: &[f32]| {
        row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
    };
    let mut window = reference.window(&state, prompt, LogitRows::All).unwrap();
    reference.commit(&mut state, &window, prompt.len()).unwrap();
    let mut answer = vec![arg_max(window.logits_of(prompt.len() - 1).unwrap())];
    while answer.len() < n {
        let token = [answer[answer.len() - 1]];
        window = reference.window(&state, &token, LogitRows::Last).unwrap();
        reference.commit(&mut state, &window, 1).unwrap();
        answer.push(arg_max(window.logits_of(0).unwrap()));
    }
    answer
}

/// A prompt of `n` tokens of the vocabulary, from a seed.
fn prompt(n: usize, seed: u64) -> Vec<u32> {
    Shape::small().tokens(n, seed).into_iter().map(|t| t % 80 + 10).collect()
}

/// A chat template with the structure of the Qwen3.x ones as far as thinking
/// goes: the reasoning of an assistant turn between think tags, and a prompt
/// that opens the block or closes an empty one.
const THINKING_TEMPLATE: &str = r#"
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- else %}
        {%- set reasoning = (message.reasoning_content if message.reasoning_content is string else '')|trim %}
        {{- '<|im_start|>assistant\n<think>\n' + reasoning + '\n</think>\n\n' + message.content + '<|im_end|>\n' }}
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

/// What a model with `template` says about its chat when it is set up with
/// `chat`.
fn declared(template: Option<&str>, chat: ChatOptions) -> Option<ChatFormat> {
    let devices = open_devices(Placement::One);
    let gguf = Gguf::open(gguf_path(1)).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let opts = options(Placement::One, n_layer, 16, MAX_ROWS, MAX_ROWS);
    Qwen4Exp::new(gguf, byte_tokenizer(template), "made", &devices, opts, CELLS)
        .unwrap()
        .with_chat_options(chat)
        .chat_format()
}

#[test]
fn the_model_declares_what_its_template_says_about_thinking() {
    if !enabled() {
        return;
    }
    let ChatFormat { thinking, .. } = declared(Some(THINKING_TEMPLATE), ChatOptions::default())
        .expect("a model with a template declares its format");
    let format = thinking.expect("the template thinks");
    assert_eq!(format.start, "<think>\n");
    assert_eq!(format.end, "\n</think>\n\n");
    assert_eq!(format.prompt_on, "<think>\n");
    assert_eq!(format.prompt_off, "<think>\n\n</think>\n\n");
    assert!(format.on_by_default && format.switchable());

    // Whether the model thinks when a request does not say is what the server
    // started it with.
    let off = ChatOptions {
        variables: vec![("enable_thinking".to_owned(), Json::Bool(false))],
        ..ChatOptions::default()
    };
    let format = declared(Some(THINKING_TEMPLATE), off).unwrap().thinking.unwrap();
    assert!(!format.on_by_default);

    // A file without a template declares no thinking.
    assert_eq!(declared(None, ChatOptions::default()).unwrap().thinking, None);
}

#[test]
fn the_runtime_answers_as_the_session_does_in_every_placement() {
    if !enabled() {
        return;
    }
    let prompt = prompt(21, 3);
    for placement in placements() {
        let want = session_answer(1, placement, &prompt, 24);
        let (runtime, model) = start(1, placement, 16, RuntimeConfig::default());
        // The first answer comes from experts that are fetched: the same
        // text up to a near tie. The settled one is the session's.
        let first = ask(&runtime, 1, &prompt, 24);
        warm(&model);
        let got = ask(&runtime, 2, &prompt, 24);
        assert_nothing_missed(&model, "the answer with filled tiers");
        runtime.shutdown();
        eprintln!(
            "RES {placement:?}: {} tokens equal the session's; the first answer agrees in {} of them",
            got.len(),
            first.iter().zip(&want).filter(|(a, b)| a == b).count()
        );
        assert_eq!(got, want, "{placement:?}: the runtime says something else than the session");
        assert_eq!(first[..5], want[..5], "{placement:?}: the answer with fetched experts");
    }
}

/// A prompt read in slices of up to 64 rows, on the many-row kernels, with every expert
/// fetched (none has a slot when the model starts), in every placement: the answer starts as
/// the CPU reference's does. (Later tokens of a made-up model's answer follow near ties, which
/// every difference of arithmetic flips; the numbers printed say how far each agrees.)
#[test]
fn a_prompt_read_in_wide_steps_gives_the_reference_answer() {
    if !enabled() {
        return;
    }
    let prompt = prompt(200, 5);
    let reference = reference_answer(1, &prompt, 16);
    for placement in placements() {
        let eight = session_answer(1, placement, &prompt, 16);
        let (runtime, model) = start_rows(1, placement, 16, RuntimeConfig::default(), WIDE_ROWS);
        let got = ask(&runtime, 1, &prompt, 16);
        let wide = model.wide_stats().expect("the parts are open");
        runtime.shutdown();
        let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
        eprintln!(
            "WIDE {placement:?}: {} tokens as the reference's before the first difference \
             (steps of eight rows: {}); {} layers of wide steps picked {} experts, {} without a \
             slot, in {} more rounds",
            leading(&got, &reference),
            leading(&eight, &reference),
            wide.layers,
            wide.picked,
            wide.missing,
            wide.extra_rounds
        );
        assert!(wide.layers > 0, "{placement:?}: the prompt ran in no wide step");
        assert!(wide.missing > 0, "{placement:?}: no expert was fetched");
        assert!(leading(&got, &reference) >= 2, "{placement:?}: {got:?} against {reference:?}");
    }
}

#[test]
fn the_samplers_settings_of_a_request_reach_this_model_too() {
    if !enabled() {
        return;
    }
    let prompt = prompt(21, 3);
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    for placement in placements() {
        let (runtime, model) = start(1, placement, 16, RuntimeConfig::default());
        warm(&model);
        let (plain, none) = ask_sampled(&runtime, 1, &prompt, 24, greedy.clone(), true);
        assert!(none.is_empty(), "{placement:?}: probabilities nobody asked for");
        // The model's own probabilities come with every token of the reply,
        // and the most probable alternative of a greedy reply is its token.
        // The request says no drafts, which this model does not make anyway:
        // the two work together.
        let asking = Sampling {
            logprobs: Some(LogprobRequest { top: 3, after_chain: false }),
            ..greedy.clone()
        };
        let (again, probs) = ask_sampled(&runtime, 2, &prompt, 24, asking, false);
        assert_eq!(again, plain, "{placement:?}: asking for probabilities changed the reply");
        assert_eq!(probs.len(), plain.len(), "{placement:?}");
        for (token, said) in plain.iter().zip(&probs) {
            assert_eq!(said.token, *token, "{placement:?}");
            assert_eq!(said.alternatives[0].token, *token, "{placement:?}");
            assert!(said.logprob <= 0.0 && said.alternatives.len() == 3, "{placement:?}");
        }
        // A token that must not be picked is not, and the best of the rest is.
        // The probabilities are the model's own, so the banned token is still
        // the most probable and the one picked is the next.
        let banned = Sampling {
            logit_bias: vec![LogitBias { key: BiasKey::Token(plain[0]), bias: None }],
            logprobs: Some(LogprobRequest { top: 2, after_chain: false }),
            ..greedy.clone()
        };
        let (after_ban, probs) = ask_sampled(&runtime, 3, &prompt, 24, banned, true);
        assert_ne!(after_ban[0], plain[0], "{placement:?}: the ban");
        assert_eq!(probs[0].alternatives[0].token, plain[0], "{placement:?}");
        assert_eq!(after_ban[0], probs[0].alternatives[1].token, "{placement:?}");
        assert_nothing_missed(&model, "the answers with filled tiers");
        runtime.shutdown();
    }
}

#[test]
fn a_model_with_no_drafter_of_its_own_is_not_drafted_for_even_from_a_text_that_repeats() {
    if !enabled() {
        return;
    }
    for placement in placements() {
        // The model has no draft head built yet, and the lookup in earlier
        // text takes part only next to a model drafter, never alone. The
        // text repeats, so that the lookup would have drafts to offer, and
        // the runtime has the room for windows of eight rows all the same.
        let mut prompt = prompt(12, 5);
        prompt.extend_from_within(..);
        prompt.extend_from_within(..12);
        let config = RuntimeConfig { window_rows: 8, ..RuntimeConfig::default() };
        let (runtime, model) = start(2, placement, 16, config);
        warm(&model);
        let answer = ask(&runtime, 1, &prompt, 160);
        assert_nothing_missed(&model, "an answer with filled tiers");
        let counters = runtime.snapshot().expect("the runtime is running").counters;
        runtime.shutdown();
        eprintln!(
            "RES {placement:?}: {} tokens, draft rows {}, drafts accepted {}, rejected {}",
            answer.len(),
            counters.draft_rows,
            counters.drafts_accepted,
            counters.drafts_rejected
        );
        assert!(answer.len() > 12, "{placement:?}: the answer is {} tokens", answer.len());
        assert_eq!(counters.draft_rows, 0, "{placement:?}: draft rows in a step");
        assert_eq!(counters.drafts_accepted, 0, "{placement:?}");
        assert_eq!(counters.drafts_rejected, 0, "{placement:?}");
    }
}

#[test]
fn two_sequences_at_once_say_what_each_says_alone() {
    if !enabled() {
        return;
    }
    let (a, b) = (prompt(19, 7), prompt(23, 8));
    for placement in placements() {
        let (runtime, model) = start(3, placement, 16, RuntimeConfig::default());
        warm(&model);
        let alone = (ask(&runtime, 1, &a, 30), ask(&runtime, 2, &b, 30));
        let (rx_a, rx_b) = (submit(&runtime, 3, &a, 30), submit(&runtime, 4, &b, 30));
        let together = (
            rx_a.recv_timeout(Duration::from_secs(300)).unwrap().unwrap(),
            rx_b.recv_timeout(Duration::from_secs(300)).unwrap().unwrap(),
        );
        assert_nothing_missed(&model, "two sequences with filled tiers");
        let counters = runtime.snapshot().map(|s| s.counters);
        runtime.shutdown();
        if let Some(c) = counters {
            eprintln!(
                "RES {placement:?}: {} steps, {} mixed, {} rows",
                c.steps, c.mixed_steps, c.rows
            );
        }
        assert_eq!(
            together, alone,
            "{placement:?}: a sequence says something else next to another"
        );
    }
}

#[test]
fn a_request_that_gives_way_comes_back_with_the_same_answer() {
    if !enabled() {
        return;
    }
    // Two requests of 100 prompt tokens and 300 generated ones need four
    // pages each. The accounting charges a newcomer for its prompt and one
    // page of output at least and expects the output it was told to expect
    // (almost none), so both start in a pool of six pages, and when their
    // output grows one of them has to give way.
    let (a, b) = (prompt(100, 11), prompt(100, 12));
    for placement in placements() {
        let config =
            RuntimeConfig { output_ratio: 0.0, output_clip: 8, ..RuntimeConfig::default() };
        let (runtime, model) = start(4, placement, 6, config);
        warm(&model);
        let (rx_a, rx_b) = (submit(&runtime, 3, &a, 300), submit(&runtime, 4, &b, 300));
        let together = (
            rx_a.recv_timeout(Duration::from_secs(600)).unwrap().unwrap(),
            rx_b.recv_timeout(Duration::from_secs(600)).unwrap().unwrap(),
        );
        let counters = runtime.snapshot().map(|s| s.counters);
        let suspended = counters.as_ref().map_or(0, |c| c.requests_suspended);
        let alone = (ask(&runtime, 1, &a, 300), ask(&runtime, 2, &b, 300));
        assert_nothing_missed(&model, "requests with filled tiers");
        runtime.shutdown();
        eprintln!("RES {placement:?}: {suspended} requests gave way");
        assert!(suspended >= 1, "{placement:?}: nobody gave way: {counters:?}");
        assert_eq!(together, alone, "{placement:?}: a request that gave way says something else");
    }
}

#[test]
fn a_cancelled_request_frees_its_place_for_the_next() {
    if !enabled() {
        return;
    }
    let (a, b) = (prompt(30, 21), prompt(25, 22));
    for placement in placements() {
        let (runtime, model) = start(5, placement, 16, RuntimeConfig::default());
        warm(&model);
        let want = ask(&runtime, 1, &b, 20);
        let (tx, rx) = mpsc::channel();
        let sink = EventSink::new(move |event| {
            if let Event::Done { finish, .. } = event {
                let _ = tx.send(finish);
            }
            true
        });
        let words: Vec<String> = a.iter().map(|t| format!("t{t}")).collect();
        runtime
            .submit(
                GenerateRequest {
                    model: "made".into(),
                    prompt: Prompt::Text(words.join(" ")),
                    sampling: Sampling::default(),
                    max_tokens: Some(400),
                    stop: Vec::new(),
                    draft: true,
                    structure: None,
                    id: RequestId(2),
                },
                sink,
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(200));
        runtime.cancel(RequestId(2));
        let finish = rx.recv_timeout(Duration::from_secs(60)).expect("the cancelled request ends");
        assert_eq!(format!("{finish:?}"), "Cancelled", "{placement:?}");
        // Its slot and its pages are free for the next request.
        assert_eq!(ask(&runtime, 3, &b, 20), want, "{placement:?}");
        runtime.shutdown();
    }
}
