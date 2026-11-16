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
use ares_model::qwen4exp::{
    Config, DraftState, DraftWeights, LogitRows, Reference, SequenceState, Weights,
};
use ares_runtime::expert_store::Serve;
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, GpuConfig, LinkSpec, Machine, Model, ModelError,
    PrefixCache, Rule, Runtime, RuntimeConfig, Split,
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
    gguf_path_of(seed, GgmlType::IQ2_S, GgmlType::Q2_0)
}

/// The file of the made-up model with gate and up of the experts as `gate_ty` and down as
/// `down_ty` (an inner width of 64, or of the type's block where that is larger).
fn gguf_path_of(seed: u64, gate_ty: GgmlType, down_ty: GgmlType) -> PathBuf {
    let mut synthetic = model(seed);
    synthetic.shape.expert_ff = (down_ty.block_values() as usize).max(64);
    let name = if (gate_ty, down_ty) == (GgmlType::IQ2_S, GgmlType::Q2_0) {
        format!("q4rt-{seed}.gguf")
    } else {
        format!("q4rt-{seed}-{gate_ty}-{down_ty}.gguf")
    };
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, synthetic.bytes_with_experts(gate_ty, down_ty))
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
struct Made(Arc<Qwen4Exp>, Cache);

/// Whether the engine over [`Made`] has the model's prefix cache.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cache {
    On,
    Off,
}

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

    fn drafters(&self) -> Vec<ares_runtime::Drafter> {
        self.0.drafters()
    }
    fn attention_span(&self) -> u32 {
        self.0.attention_span()
    }
    fn prefix_cache(&self) -> Option<PrefixCache> {
        self.0.prefix_cache().filter(|_| self.1 == Cache::On)
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
    start_at(&gguf_path(seed), placement, pages, config, rows)
}

/// [`start`] for an engine whose tier of experts in RAM follows `machine`, in
/// pieces of `piece` bytes.
fn start_following(
    seed: u64,
    placement: Placement,
    pages: u32,
    machine: &Arc<Machine>,
    piece: usize,
) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let mut opts = options(placement, n_layer, pages, MAX_ROWS, MAX_ROWS);
    opts.ram_piece = Some(piece);
    let inner =
        Arc::new(Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap());
    let config =
        RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..RuntimeConfig::default() };
    let model: Arc<dyn Model> = Arc::new(Made(inner.clone(), Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    let runtime = Runtime::start_gpu_following(model, devices, config, gpu, Some(machine.clone()))
        .expect("the runtime starts");
    (runtime, inner)
}

/// The resident bytes of this process, as the kernel counts them.
fn resident() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let line = status.lines().find_map(|line| line.strip_prefix("VmRSS:")).unwrap_or("0 kB");
    line.trim().trim_end_matches("kB").trim().parse::<u64>().unwrap_or(0) * 1024
}

/// [`start_rows`] for the model file at `path`.
fn start_at(
    path: &std::path::Path,
    placement: Placement,
    pages: u32,
    config: RuntimeConfig,
    rows: u32,
) -> (Runtime, Arc<Qwen4Exp>) {
    start_with(path, None, placement, pages, config, rows, Cache::On)
}

/// The file of a made-up draft block for the made-up model of `seed`, its
/// experts in the types of the model's.
fn draft_path(seed: u64) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4rt-{seed}-draft.gguf"));
    std::fs::write(&path, model(seed).draft_bytes(Some((GgmlType::IQ2_S, GgmlType::Q2_0))))
        .expect("the test directory is writable");
    path
}

/// [`start`] for a model that drafts with its draft block: windows of up to
/// eight rows.
fn start_drafting(seed: u64, placement: Placement, pages: u32) -> (Runtime, Arc<Qwen4Exp>) {
    let config = RuntimeConfig { window_rows: 8, ..RuntimeConfig::default() };
    start_with(
        &gguf_path(seed),
        Some(&draft_path(seed)),
        placement,
        pages,
        config,
        MAX_ROWS,
        Cache::On,
    )
}

/// [`start_drafting`] without the prices the conductor measures at the
/// start: windows are then chosen by what they are expected to bring alone,
/// so that a block whose drafts are never right goes on drafting whatever a
/// link costs on the day.
fn start_drafting_unpriced(
    seed: u64,
    placement: Placement,
    pages: u32,
) -> (Runtime, Arc<Qwen4Exp>) {
    let config = RuntimeConfig { window_rows: 8, warm_up: false, ..RuntimeConfig::default() };
    start_with(
        &gguf_path(seed),
        Some(&draft_path(seed)),
        placement,
        pages,
        config,
        MAX_ROWS,
        Cache::On,
    )
}

/// [`start_at`], with the draft block of the file `draft` when there is one.
fn start_with(
    path: &std::path::Path,
    draft: Option<&std::path::Path>,
    placement: Placement,
    pages: u32,
    config: RuntimeConfig,
    rows: u32,
    cache: Cache,
) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(path).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let opts = options(placement, n_layer, pages, MAX_ROWS, rows);
    let mut inner =
        Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap();
    if let Some(draft) = draft {
        inner = inner.with_draft(Gguf::open(draft).unwrap()).expect("the draft block fits");
    }
    let inner = Arc::new(inner);
    let config = RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..config };
    let model: Arc<dyn Model> = Arc::new(Made(inner.clone(), cache));
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
    if let Some((vram_slots, _)) = model.draft_slots() {
        assert!(vram_slots >= model.config().experts.count, "every expert of the draft block");
    }
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
    reference_answer_at(&gguf_path(seed), prompt, n)
}

/// [`reference_answer`] for the model file at `path`.
fn reference_answer_at(path: &std::path::Path, prompt: &[u32], n: usize) -> Vec<u32> {
    let gguf = Gguf::open(path).unwrap();
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
        // Without the prefix cache: with it, the second answer would read the
        // cells the first one computed with experts that were fetched, the
        // CPU's arithmetic for the missed ones among them.
        let path = gguf_path(1);
        let config = RuntimeConfig::default();
        let (runtime, model) = start_with(&path, None, placement, 16, config, MAX_ROWS, Cache::Off);
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

/// Models whose experts are stored in other weight types than the real file's, through the
/// runtime in every placement: a prompt read in wide steps (the many-row kernels of the
/// types, with experts fetched) and the tokens after it (the decode kernels), against the
/// CPU reference of the model.
#[test]
fn models_of_other_expert_types_answer_as_the_reference_in_every_placement() {
    if !enabled() {
        return;
    }
    let prompt = prompt(200, 5);
    for (gate_ty, down_ty) in [
        (GgmlType::Q4_K, GgmlType::Q8_0),
        (GgmlType::BF16, GgmlType::Q4_0),
        (GgmlType::IQ3_S, GgmlType::IQ4_XS),
        (GgmlType::Q5_1, GgmlType::TQ2_0),
    ] {
        let path = gguf_path_of(1, gate_ty, down_ty);
        let reference = reference_answer_at(&path, &prompt, 16);
        for placement in placements() {
            let (runtime, model) =
                start_at(&path, placement, 16, RuntimeConfig::default(), WIDE_ROWS);
            let got = ask(&runtime, 1, &prompt, 16);
            let wide = model.wide_stats().expect("the parts are open");
            runtime.shutdown();
            let leading = got.iter().zip(&reference).take_while(|(x, y)| x == y).count();
            eprintln!(
                "TYPES {gate_ty} and {down_ty}, {placement:?}: {leading} tokens as the \
                 reference's before the first difference; {} layers of wide steps picked {} \
                 experts, {} without a slot",
                wide.layers, wide.picked, wide.missing
            );
            assert!(wide.layers > 0, "{placement:?}: the prompt ran in no wide step");
            assert!(
                leading >= 2,
                "{gate_ty} and {down_ty}, {placement:?}: {got:?} against {reference:?}"
            );
        }
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
        // The model has no draft head built yet, and the n-gram drafter
        // takes part only next to a model drafter, never alone. The
        // text repeats, so that the n-gram drafter would have drafts to offer, and
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

/// The tokens `prompt` and what a request wrote after it, as the CPU
/// reference of the model and of its draft block see them: the residual
/// after the last layer of every cell, and the block's cells that follow.
struct Followed {
    tokens: Vec<u32>,
    trunk: Vec<f32>,
    wide: usize,
}

/// The chains the draft block of the made-up model `seed` made for a request
/// that read `prompt` and wrote `answer`, against the CPU reference's chains
/// from the same cells: per link, how many were compared (a link is compared
/// when the links before it agreed) and how many agreed.
fn chains_against_the_reference(
    seed: u64,
    chains: &[ares_model::qwen4exp::device::Drafted],
    prompt: &[u32],
    answer: &[u32],
) -> ([usize; 8], [usize; 8]) {
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let file = Gguf::open(draft_path(seed)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let head = DraftWeights::load(&file, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let tokens: Vec<u32> = prompt.iter().chain(answer).copied().collect();
    let trunk = reference.window(&SequenceState::new(&c), &tokens, LogitRows::All).unwrap();
    // How far the model's own picks on the device are the reference's from
    // the same tokens: what near ties do to a made-up model.
    let arg_max = |row: &[f32]| {
        row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
    };
    let own = (prompt.len() - 1..tokens.len() - 1)
        .filter(|&i| arg_max(trunk.logits_of(i).unwrap()) == tokens[i + 1])
        .count();
    eprintln!("RES the model's own picks are the reference's in {own} of {}", answer.len());
    let f = Followed { tokens, trunk: trunk.trunk_out, wide: c.wide() };
    let cells = f.tokens.len() - 1;
    let follow = reference
        .draft_window(
            &head,
            &DraftState::new(),
            &f.trunk[..cells * f.wide],
            &f.tokens[1..],
            LogitRows::None,
        )
        .unwrap();
    let (mut compared, mut agreed) = ([0usize; 8], [0usize; 8]);
    for chain in chains {
        // The window started at the newest token's cell; the chain's first
        // link is the cell before it.
        let newest = chain.position as usize - 1;
        if newest == 0 || newest >= f.tokens.len() || f.tokens[newest] != chain.token {
            continue;
        }
        let mut state = DraftState::new();
        reference.draft_commit(&mut state, &follow, newest - 1).unwrap();
        let row = &f.trunk[(newest - 1) * f.wide..newest * f.wide];
        let want =
            reference.draft_chain(&head, &state, row, chain.token, chain.drafts.len()).unwrap();
        for (link, (got, want)) in chain.drafts.iter().zip(&want).enumerate() {
            compared[link] += 1;
            if got != want {
                eprintln!(
                    "RES a chain from cell {} parts from the reference's at link {link} of {}",
                    newest - 1,
                    chain.drafts.len()
                );
                break;
            }
            agreed[link] += 1;
        }
    }
    (compared, agreed)
}

/// The model drafts with its own draft block in every placement, and the
/// block on the device drafts what the CPU reference's chain drafts from the
/// same cells: its input from the model's residual and the paired token, its
/// keys and values of the cells the model has passed, its experts through
/// the store, its own final mixer and the model's head, link after link.
#[test]
fn the_draft_block_drafts_what_the_reference_drafts_in_every_placement() {
    if !enabled() {
        return;
    }
    let prompt = prompt(21, 3);
    for placement in placements() {
        let (runtime, model) = start_drafting_unpriced(1, placement, 16);
        warm(&model);
        model.take_drafted();
        let answer = ask(&runtime, 1, &prompt, 40);
        assert_nothing_missed(&model, "an answer with drafts and filled tiers");
        let chains = model.take_drafted();
        let counters = runtime.snapshot().expect("the runtime is running").counters;
        let report = model.draft_report().expect("the model has a draft block");
        let times = model.draft_times();
        runtime.shutdown();
        let (compared, agreed) = chains_against_the_reference(1, &chains, &prompt, &answer);
        eprintln!(
            "RES {placement:?}: {} chains of {} links in all; against the reference per link \
             {agreed:?} of {compared:?}; draft rows {}, accepted {}, rejected {}; the block on \
             part {}: weights {} B, pools {} B, buffers {} B; {:.2} ms a link",
            chains.len(),
            counters.own_draft_links,
            counters.draft_rows,
            counters.drafts_accepted,
            counters.drafts_rejected,
            report.index,
            report.weights,
            report.pools,
            report.buffers,
            times.chain_ms / times.links.max(1) as f64,
        );
        assert!(counters.own_draft_links > 0, "{placement:?}: the block drafted nothing");
        // A made-up block is never right, so the policy stops asking it after
        // a chain or a few: what there is must be the reference's. A made-up
        // model has near ties that the device's arithmetic flips, so one
        // chain in five may part; a block that is wired wrongly agrees by
        // chance, once in a hundred.
        assert!(compared[0] >= 1, "{placement:?}: no chain to compare");
        assert!(
            agreed[0] + compared[0] / 5 + usize::from(compared[0] < 5 && compared[0] > 1)
                >= compared[0],
            "{placement:?}: first links agree in {} of {}",
            agreed[0],
            compared[0]
        );
    }
}

/// Drafts change no token: with the draft block a reply is what it is
/// without, greedy and sampled with a seed, in every placement. The made-up
/// block guesses badly, so most of its drafts are rejected and rolled back;
/// the text repeats and the request is asked again, so that the n-gram drafter in
/// earlier text joins and has drafts that are accepted, which the block then
/// has to follow. Two sequences that share their steps write what each
/// writes alone up to a near tie, with drafts and without.
#[test]
fn drafts_change_no_token_in_any_placement() {
    if !enabled() {
        return;
    }
    let mut repeats = prompt(12, 5);
    repeats.extend_from_within(..);
    repeats.extend_from_within(..12);
    let other = prompt(23, 8);
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    let seeded = Sampling { temperature: Some(0.9), seed: Some(11), ..Sampling::default() };
    let mut failures = Vec::new();
    for placement in placements() {
        // Without the block.
        let (runtime, model) = start(2, placement, 16, RuntimeConfig::default());
        warm(&model);
        let plain = ask(&runtime, 1, &repeats, 120);
        let plain_other = ask(&runtime, 2, &other, 60);
        let (plain_seeded, _) = ask_sampled(&runtime, 3, &repeats, 60, seeded.clone(), true);
        // Two sequences that share their steps, still without drafts: what a
        // sequence says next to another one is its own text up to near ties
        // (a row's sums are not the same bits in every shape of step, and a
        // made-up model amplifies the difference).
        let (rx_a, rx_b) = (submit(&runtime, 4, &repeats, 120), submit(&runtime, 5, &other, 60));
        let plain_together = (
            rx_a.recv_timeout(Duration::from_secs(300)).unwrap().unwrap(),
            rx_b.recv_timeout(Duration::from_secs(300)).unwrap().unwrap(),
        );
        assert_nothing_missed(&model, "the answers without drafts");
        runtime.shutdown();
        drop(model);

        let (runtime, model) = start_drafting(2, placement, 16);
        warm(&model);
        let first = ask(&runtime, 1, &repeats, 120);
        // The same again: the n-gram drafter has the whole answer to offer.
        let again = ask(&runtime, 2, &repeats, 120);
        let (sampled, _) = ask_sampled(&runtime, 3, &repeats, 60, seeded.clone(), true);
        let (undrafted, _) = ask_sampled(&runtime, 4, &repeats, 120, greedy.clone(), false);
        let (rx_a, rx_b) = (submit(&runtime, 5, &repeats, 120), submit(&runtime, 6, &other, 60));
        let together = (
            rx_a.recv_timeout(Duration::from_secs(300)).unwrap().unwrap(),
            rx_b.recv_timeout(Duration::from_secs(300)).unwrap().unwrap(),
        );
        assert_nothing_missed(&model, "the answers with drafts and filled tiers");
        let counters = runtime.snapshot().expect("the runtime is running").counters;
        let times = model.draft_times();
        runtime.shutdown();
        eprintln!(
            "RES {placement:?}: {} tokens; draft rows {}, accepted {}, rejected {}; the block \
             made {} links in {} steps, verified per link {:?}, accepted {:?}; {} rows only \
             wrote keys",
            plain.len(),
            counters.draft_rows,
            counters.drafts_accepted,
            counters.drafts_rejected,
            counters.own_draft_links,
            counters.own_draft_steps,
            times.offered,
            times.accepted,
            times.key_rows,
        );
        assert!(plain.len() > 12, "{placement:?}: the answer is {} tokens", plain.len());
        assert!(counters.own_draft_links > 0, "{placement:?}: the block drafted nothing");
        assert!(counters.drafts_accepted > 0, "{placement:?}: no draft was accepted");
        assert!(counters.drafts_rejected > 0, "{placement:?}: no draft was rejected");
        // Alone, a sequence writes with drafts what it writes without, token
        // for token.
        for (what, got, want) in [
            ("drafts changed a greedy reply", &first, &plain),
            ("accepted drafts changed a greedy reply", &again, &plain),
            ("a request without drafts", &undrafted, &plain),
            ("drafts changed a sampled reply", &sampled, &plain_seeded),
        ] {
            if got != want {
                failures.push(format!("{placement:?}: {what}: {got:?} and not {want:?}"));
            }
        }
        // Next to another sequence it does up to a near tie, with drafts as
        // without: where two replies part, the reference has the two tokens
        // within a hundredth of the spread of its logits.
        for (what, prompt, got, want) in [
            ("two sequences without drafts", &repeats, &plain_together.0, &plain),
            ("the other of two without drafts", &other, &plain_together.1, &plain_other),
            ("next to another sequence", &repeats, &together.0, &plain),
            ("the other sequence", &other, &together.1, &plain_other),
        ] {
            let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) else {
                assert_eq!(got.len(), want.len(), "{placement:?}: {what}");
                continue;
            };
            let [a, b, largest, spread] = reference_gap(2, prompt, &want[..at], got[at], want[at]);
            eprintln!(
                "RES {placement:?}: {what}: token {at} is {} and not {}: the reference has \
                 their logits at {a} and {b}, the largest at {largest}, a spread of {spread}",
                got[at], want[at]
            );
            if (a - b).abs() > 0.01 * spread {
                failures.push(format!("{placement:?}: {what}: token {at} is no near tie"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// What the CPU reference says of the row behind `prompt` and `written`: the
/// logits of the tokens `a` and `b`, the largest logit and the standard
/// deviation of all of them.
fn reference_gap(seed: u64, prompt: &[u32], written: &[u32], a: u32, b: u32) -> [f32; 4] {
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let tokens: Vec<u32> = prompt.iter().chain(written).copied().collect();
    let window = reference.window(&SequenceState::new(&c), &tokens, LogitRows::Last).unwrap();
    let logits = window.logits_of(tokens.len() - 1).unwrap();
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    let spread =
        (logits.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / logits.len() as f32).sqrt();
    let largest = logits.iter().copied().fold(f32::MIN, f32::max);
    [logits[a as usize], logits[b as usize], largest, spread]
}

/// `<|im_start|>user\nErkläre in drei Sätzen, warum der Himmel blau
/// ist.<|im_end|>\n<|im_start|>assistant\n`, as tokens of Flash-Next, and the
/// four tokens the CPU reference and Strata continue with
/// (`tests/qwen4exp_real.rs`).
const FLASH_PROMPT: [u32; 22] = [
    248045, 846, 198, 18871, 10248, 45125, 303, 55593, 326, 179212, 11, 183857, 2607, 199908,
    242880, 5810, 13, 248046, 198, 248045, 74455, 198,
];
const FLASH_CONTINUATION: [u32; 4] = [248068, 198, 760, 1156];

/// The runtime over the real model on both cards (or the one there is), with
/// every miss uploaded and none computed by the CPU, whose arithmetic is not
/// the device's: what a row computes then does not depend on where its
/// experts were found. With `draft` the model drafts with that block, in
/// windows of four rows whose price is learned from the steps that run.
fn start_flash(
    path: &std::path::Path,
    draft: Option<&std::path::Path>,
) -> (Runtime, Arc<Qwen4Exp>) {
    start_flash_with(path, draft, None)
}

/// [`start_flash`] with a DFlash drafter as well, in windows of eight rows.
fn start_flash_with(
    path: &std::path::Path,
    draft: Option<&std::path::Path>,
    dflash: Option<Arc<ares_model::dflash::File>>,
) -> (Runtime, Arc<Qwen4Exp>) {
    let placement = if device_count().expect("the CUDA driver") >= 2 {
        Placement::Shared
    } else {
        Placement::One
    };
    let devices = open_devices(placement);
    let gguf = Gguf::open(path).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let mut opts = options(placement, n_layer, 16, MAX_ROWS, MAX_ROWS);
    opts.serve = Serve::Stream;
    opts.reserve = None;
    opts.spread = if placement == Placement::Shared { Spread::Auto } else { Spread::Layers };
    let mut inner =
        Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap();
    if let Some(draft) = draft {
        inner = inner.with_draft(Gguf::open(draft).unwrap()).expect("the draft block fits");
    }
    let window = match (&dflash, draft) {
        (Some(_), _) => 8,
        (None, Some(_)) => 4,
        (None, None) => 1,
    };
    if let Some(file) = dflash {
        inner = inner.with_dflash(file).expect("the DFlash drafter fits");
    }
    let inner = Arc::new(inner);
    let config = RuntimeConfig {
        pages: 16,
        cells_per_page: 128,
        max_slots: SLOTS,
        window_rows: window,
        warm_up: false,
        ..RuntimeConfig::default()
    };
    let model: Arc<dyn Model> = Arc::new(Made(inner.clone(), Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    // A small tier in RAM: what has no slot in VRAM is read from the file.
    gpu.ram_limit += 1 << 30;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// On the real model drafts change no token: Flash-Next writes with its draft
/// block, token for token, what it writes without drafts, greedy and sampled
/// with a seed, and the block's drafts are accepted. The CPU serves no miss
/// (see [`start_flash`]), and both replies come from one engine: two engines
/// on two cards each measure at their start how the cards share every layer
/// (the time of a read of a stream mixer on each), the share differs from
/// start to start (0.733 to 0.751 of the key heads on the faster card in eight
/// starts), and with it how the sums of a layer are cut between the cards, so
/// the comparison would be one of two arithmetics, which a near tie tells
/// apart. Needs `ARES_TEST_QWEN4EXP_MODEL` and `ARES_TEST_QWEN4EXP_DRAFT`.
#[test]
fn flash_next_writes_with_drafts_what_it_writes_without() {
    if !enabled() {
        return;
    }
    let (Some(path), Some(draft)) = (
        std::env::var_os("ARES_TEST_QWEN4EXP_MODEL").map(PathBuf::from),
        std::env::var_os("ARES_TEST_QWEN4EXP_DRAFT").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipped: set ARES_TEST_QWEN4EXP_MODEL to the first part of Flash-Next and \
             ARES_TEST_QWEN4EXP_DRAFT to the file of its draft block"
        );
        return;
    };
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    let seeded = Sampling { temperature: Some(0.8), seed: Some(5), ..Sampling::default() };
    let (runtime, model) = start_flash(&path, Some(&draft));
    // The replies with drafts first: a reply that repeats one the engine has
    // written is drafted by the n-gram drafter alone.
    let drafted = ask(&runtime, 1, &FLASH_PROMPT, 64);
    let (sampled, _) = ask_sampled(&runtime, 2, &FLASH_PROMPT, 48, seeded.clone(), true);
    let counters = runtime.snapshot().expect("the runtime is running").counters;
    let times = model.draft_times();
    let (plain, _) = ask_sampled(&runtime, 3, &FLASH_PROMPT, 64, greedy, false);
    let (plain_seeded, _) = ask_sampled(&runtime, 4, &FLASH_PROMPT, 48, seeded, false);
    let undrafted = runtime.snapshot().expect("the runtime is running").counters.draft_rows
        - counters.draft_rows;
    let stats = model.take_store_stats().expect("the parts are open");
    runtime.shutdown();
    eprintln!(
        "RES Flash-Next: {} and {} tokens; draft rows {}, accepted {}, rejected {}; the block \
         made {} links in {} steps, {:.2} ms a link; verified per link {:?}, accepted {:?}; \
         prefix cache: {} hits, {} rows replayed",
        plain.len(),
        plain_seeded.len(),
        counters.draft_rows,
        counters.drafts_accepted,
        counters.drafts_rejected,
        times.links,
        times.chain_steps,
        times.chain_ms / times.links.max(1) as f64,
        times.offered,
        times.accepted,
        counters.prefix_hits,
        counters.replay_rows,
    );
    assert_eq!(stats.cpu_experts, 0, "the CPU computed experts: {stats:?}");
    assert_eq!(plain[..4], FLASH_CONTINUATION, "the reference's first tokens");
    assert_eq!(undrafted, 0, "the requests without drafts were drafted for");
    assert!(times.accepted[0] > 0, "none of the block's drafts was accepted: {times:?}");
    assert!(counters.drafts_rejected > 0, "no draft was rejected");
    assert_eq!(drafted, plain, "drafts changed a greedy reply of Flash-Next");
    assert_eq!(sampled, plain_seeded, "drafts changed a sampled reply of Flash-Next");
}

/// The same with the DFlash drafter (`ARES_TEST_DFLASH_FLASHNEXT`), which reads
/// the mean of the model's four streams: Flash-Next writes with its drafts,
/// greedy, what it writes without, on one engine, and some of them are
/// accepted.
#[test]
fn flash_next_writes_with_dflash_drafts_what_it_writes_without() {
    if !enabled() {
        return;
    }
    let (Some(path), Some(file)) = (
        std::env::var_os("ARES_TEST_QWEN4EXP_MODEL").map(PathBuf::from),
        std::env::var_os("ARES_TEST_DFLASH_FLASHNEXT").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipped: set ARES_TEST_QWEN4EXP_MODEL to the first part of Flash-Next and \
             ARES_TEST_DFLASH_FLASHNEXT to its DFlash2 file"
        );
        return;
    };
    let file = Arc::new(ares_model::dflash::File::open(&file).unwrap());
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    let (runtime, model) = start_flash_with(&path, None, Some(file));
    // The reply with drafts first, as above.
    let drafted = ask(&runtime, 1, &FLASH_PROMPT, 64);
    let counters = runtime.snapshot().expect("the runtime is running").counters;
    let (plain, _) = ask_sampled(&runtime, 2, &FLASH_PROMPT, 64, greedy, false);
    let undrafted = runtime.snapshot().expect("the runtime is running").counters.draft_rows
        - counters.draft_rows;
    let stats = model.take_store_stats().expect("the parts are open");
    runtime.shutdown();
    eprintln!(
        "RES Flash-Next with DFlash: {} tokens in {} steps; tested by position {:?}, accepted \
         {:?}; prefix cache: {} hits, {} cells cached, {} rows replayed, {} checkpoints restored",
        counters.tokens,
        counters.steps,
        &counters.drafts_tested_at[2][..7],
        &counters.drafts_accepted_at[2][..7],
        counters.prefix_hits,
        counters.cached_cells,
        counters.replay_rows,
        counters.checkpoints_restored,
    );
    assert_eq!(stats.cpu_experts, 0, "the CPU computed experts: {stats:?}");
    assert_eq!(plain[..4], FLASH_CONTINUATION, "the reference's first tokens");
    assert_eq!(undrafted, 0, "the request without drafts was drafted for");
    assert!(counters.drafts_accepted_at[2][0] > 0, "no DFlash draft was accepted");
    assert_eq!(drafted, plain, "DFlash drafts changed a greedy reply of Flash-Next");
}

/// The tier of experts in RAM under the runtime, on a machine whose memory the test sets:
/// when the machine runs short the tier gives its pieces back, the process's resident memory
/// falls by what was given (the kernel's count: page-locked memory really went back), the
/// engine answers as before, and when memory has been free for a while the tier takes its
/// pieces again.
#[test]
fn the_ram_tier_gives_its_memory_back_to_the_system_and_takes_it_again() {
    use std::sync::atomic::{AtomicU64, Ordering};
    if !enabled() {
        return;
    }
    const PIECE: usize = 1 << 20;
    let available = Arc::new(AtomicU64::new(64 << 30));
    let ms = Arc::new(AtomicU64::new(0));
    let rule = Rule {
        reserve: 64 << 20,
        speed: 0,
        latency: Duration::ZERO,
        piece: PIECE as u64,
        calm: Duration::from_secs(1),
        most: 64 << 20,
        ..Rule::default()
    };
    let machine = {
        let (available, ms) = (available.clone(), ms.clone());
        Machine::new(
            rule,
            move || Some(available.load(Ordering::SeqCst)),
            move || Duration::from_millis(ms.load(Ordering::SeqCst)),
        )
    };
    let look = || {
        ms.fetch_add(50, Ordering::SeqCst);
        machine.look().expect("the machine says what it has")
    };
    let prompt = prompt(21, 3);
    let (runtime, model) = start_following(1, Placement::One, 16, &machine, PIECE);
    ask(&runtime, 1, &prompt, 24);
    warm(&model);
    let want = ask(&runtime, 2, &prompt, 24);
    let (_, slots, whole) = model.expert_slots().expect("the parts are open");
    assert!(whole >= 8 * PIECE as u64, "a tier of several pieces: {whole}");
    let before = resident();

    // The machine runs short of memory.
    available.store(0, Ordering::SeqCst);
    for _ in 0..100 {
        if look().owed == 0 {
            break;
        }
    }
    let (_, slots_low, low) = model.expert_slots().expect("the parts are open");
    let after = resident();
    eprintln!(
        "RES the RAM tier went from {whole} bytes ({slots} slots) to {low} ({slots_low}); the \
         process's resident memory from {before} to {after}"
    );
    assert!(low <= whole / 2 && slots_low < slots, "{whole} -> {low}, {slots} -> {slots_low}");
    assert!(
        before.saturating_sub(after) + (2 << 20) >= whole - low,
        "what the tier gave went back to the system: {before} -> {after} for {}",
        whole - low
    );
    assert_eq!(ask(&runtime, 3, &prompt, 24), want, "the answer with a small tier");

    // Memory is free again: nothing is taken at once, all of it after a while.
    available.store(64 << 30, Ordering::SeqCst);
    look();
    assert_eq!(model.expert_slots().expect("open").2, low, "not before it has been free a while");
    for _ in 0..200 {
        look();
        if model.expert_slots().expect("open").2 == whole {
            break;
        }
    }
    let (_, slots_again, again) = model.expert_slots().expect("the parts are open");
    assert_eq!((slots_again, again), (slots, whole), "the tier has its pieces again");
    assert!(resident() + (2 << 20) >= after + (whole - low), "and they are locked again");
    assert_eq!(ask(&runtime, 4, &prompt, 24), want, "the answer with the tier it had");
    runtime.shutdown();
}

/// [`start`] for a model that keeps the usage profile of its experts in
/// `dir`, whose misses are all uploaded (the CPU's arithmetic is not the
/// device's, and a run that misses less would compute other experts there).
fn start_profiled(placement: Placement, dir: &std::path::Path) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(gguf_path(1)).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let mut opts = options(placement, n_layer, 16, MAX_ROWS, MAX_ROWS);
    opts.serve = Serve::Stream;
    let inner = Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS)
        .unwrap()
        .with_usage_profile(dir, Arc::new(|line| eprintln!("LOG {line}")));
    let inner = Arc::new(inner);
    let config = RuntimeConfig {
        pages: 16,
        cells_per_page: 128,
        max_slots: SLOTS,
        ..RuntimeConfig::default()
    };
    let model: Arc<dyn Model> = Arc::new(Made(inner.clone(), Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// The usage profile of the experts outlives the engine: the first engine has
/// none and writes it when it goes, the next one fills its tiers from it before
/// the first step (in every placement, both devices on every layer
/// included, where an expert's heat is its owner's) and answers as the first
/// did. What the store keeps of an idle engine changes no answer.
#[test]
fn the_usage_profile_outlives_the_engine_and_fills_the_tiers() {
    use ares_model::qwen4exp::device::{IDLE_AFTER, ProfileFound};
    if !enabled() {
        return;
    }
    let prompt = prompt(40, 5);
    for placement in placements() {
        let dir =
            PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4rt-usage-{placement:?}"));
        let _ = std::fs::remove_dir_all(&dir);
        let profiles = || std::fs::read_dir(&dir).map_or(0, |entries| entries.count());

        let (runtime, model) = start_profiled(placement, &dir);
        let report = model.serving().expect("the model keeps a profile");
        assert_eq!(report.found, ProfileFound::None, "{placement:?}");
        assert!(report.filled.is_none());
        assert_eq!(profiles(), 0, "nothing is written before a step ran");
        let first = ask(&runtime, 1, &prompt, 16);
        // Nothing runs for a while: the keeper fills what is empty.
        std::thread::sleep(IDLE_AFTER + Duration::from_millis(500));
        let usage = model.usage_stats().expect("the model keeps a profile");
        eprintln!("RES {placement:?}: after the first answer and a second of nothing: {usage:?}");
        assert_eq!(ask(&runtime, 3, &prompt, 16), first, "{placement:?}: after the idle time");
        runtime.shutdown();
        drop(runtime);
        drop(model);
        assert_eq!(profiles(), 1, "{placement:?}: the engine wrote its profile when it went");

        let (runtime, model) = start_profiled(placement, &dir);
        let report = model.serving().expect("the model keeps a profile");
        let ProfileFound::Found { known } = report.found else {
            panic!("{placement:?}: {report:?}");
        };
        let (filled, took) = report.filled.expect("the tiers were filled");
        eprintln!("RES {placement:?}: {known} experts known, filled {filled:?} in {took:?}");
        assert!(known > 0 && filled.vram > 0, "{placement:?}: {filled:?}");
        assert_eq!(filled.dropped, 0);
        let usage = model.usage_stats().unwrap();
        assert!(usage.vram.0 as u64 >= filled.vram, "{placement:?}: {usage:?}");
        model.take_store_stats();
        let again = ask(&runtime, 2, &prompt, 16);
        assert_eq!(again, first, "{placement:?}: the same answer from tiers that were filled");
        let stats = model.take_store_stats().unwrap();
        assert_eq!(stats.miss_from_disk, 0, "{placement:?}: the profile had what it used");
        runtime.shutdown();
        drop(runtime);
        drop(model);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
