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
    LogprobRequest, Prompt, RequestId, Sampling, TokenProbs,
};
use ares_core::GgmlType;
use ares_cuda::{CacheFormat, Device, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    ModelOptions, Options, Qwen4Exp, Rows, Session, SessionOptions, Spread, Wanted,
};
use ares_model::qwen4exp::synthetic::{NgramShape, Shape, Synthetic, Types};
use ares_model::qwen4exp::{
    Config, Inputs, LogitRows, MtpState, MtpWeights, Reference, SequenceState, Weights,
};
use ares_runtime::expert_store::Serve;
use ares_runtime::{
    DeviceModel, DeviceTransfer, EncodedPrompt, GpuConfig, ImageInput, ImageSpan, Logits, Machine,
    Model, ModelDrafts, ModelError, Pick, Rule, Runtime, RuntimeConfig, Spans, StepContext,
    TowerJob, TransferContext, Vision,
};
use ares_tokenizer::{ChatOptions, Json};
use common::gpu::enabled;
use common::helpers::{behind_the_best, is_near_best};
use common::model::{Wrapped, byte_tokenizer, words};

mod common;

const MAX_ROWS: u32 = 8;
/// The rows of a step that reads a slice of a prompt on the many-row kernels.
const WIDE_ROWS: u32 = 64;
const CELLS: u32 = 512;
const SLOTS: u32 = 2;

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
) -> ModelOptions {
    ModelOptions {
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
        ..ModelOptions::default()
    }
}

/// The model of the runtime with the made-up words (a prompt is `t5 t9 ...`, a token
/// prints as `t5 `), with its prefix cache or without.
fn made(inner: &Arc<Qwen4Exp>, cache: Cache) -> Wrapped {
    Wrapped::new(inner.clone()).words().cache(cache == Cache::On)
}

/// Whether the engine over [`made`] has the model's prefix cache.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cache {
    On,
    Off,
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
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
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

/// The file of a made-up MTP block for the made-up model of `seed`, its
/// experts in the types of the model's.
fn draft_path(seed: u64) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4rt-{seed}-draft.gguf"));
    std::fs::write(&path, model(seed).draft_bytes(Some((GgmlType::IQ2_S, GgmlType::Q2_0))))
        .expect("the test directory is writable");
    path
}

/// [`start`] for a model that drafts with its MTP block: windows of up to
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
    start_drafting_unpriced_rows(seed, placement, pages, MAX_ROWS)
}

/// [`start_drafting_unpriced`] with steps of up to `rows` rows.
fn start_drafting_unpriced_rows(
    seed: u64,
    placement: Placement,
    pages: u32,
    rows: u32,
) -> (Runtime, Arc<Qwen4Exp>) {
    let config = RuntimeConfig { window_rows: 8, warm_up: false, ..RuntimeConfig::default() };
    start_with(&gguf_path(seed), Some(&draft_path(seed)), placement, pages, config, rows, Cache::On)
}

/// [`start_at`], with the MTP block of the file `draft` when there is one.
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
        inner = inner.with_mtp(Gguf::open(draft).unwrap()).expect("the MTP block fits");
    }
    let inner = Arc::new(inner);
    let config = RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..config };
    let model: Arc<dyn Model> = Arc::new(made(&inner, cache));
    let mut gpu = GpuConfig::new(&config, &*model);
    // Room for the tier of experts next to what the executor needs.
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// [`start_drafting`] on two devices that share every layer, the second holding experts
/// only (`core_share` 1, the shared expert on the mixer): the role the trial at load keeps
/// on the two development cards.
fn start_drafting_experts_only(seed: u64) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(Placement::Shared);
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let mut opts = options(Placement::Shared, n_layer, 16, MAX_ROWS, MAX_ROWS);
    opts.spread =
        Spread::Tensor { mixer: Some(0), core_share: Some(1.0), shared_on_mixer: Some(true) };
    let mut inner =
        Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap();
    inner = inner.with_mtp(Gguf::open(draft_path(seed)).unwrap()).expect("the MTP block fits");
    let inner = Arc::new(inner);
    let config = RuntimeConfig {
        pages: 16,
        cells_per_page: 128,
        max_slots: SLOTS,
        window_rows: 8,
        ..RuntimeConfig::default()
    };
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// A second device that holds experts only runs draft windows: it keeps nothing for them
/// and their commit has nothing to do there (it refused them, which took the model down
/// once the trial at load chose that role, 2026-11-23). Drafts change no greedy token.
#[test]
fn a_worker_of_experts_only_runs_draft_windows() {
    if !enabled() || device_count().expect("the CUDA driver") < 2 {
        return;
    }
    let mut repeats = prompt(12, 5);
    repeats.extend_from_within(..);
    let (runtime, model) = start_drafting_experts_only(2);
    warm(&model);
    let drafted = ask(&runtime, 1, &repeats, 60);
    let again = ask(&runtime, 2, &repeats, 60);
    let (undrafted, _) = ask_sampled(
        &runtime,
        3,
        &repeats,
        60,
        Sampling { temperature: Some(0.0), ..Sampling::default() },
        false,
    );
    let counters = runtime.snapshot().expect("the runtime is running").counters;
    runtime.shutdown();
    eprintln!(
        "RES experts only: draft rows {}, accepted {}, rejected {}",
        counters.draft_rows, counters.drafts_accepted, counters.drafts_rejected
    );
    assert!(counters.draft_rows > 0, "no draft window ran");
    assert_eq!(drafted, undrafted, "drafts changed a greedy reply");
    assert_eq!(again, undrafted, "accepted drafts changed a greedy reply");
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
                Event::Progress(_) | Event::Generated(_) => {}
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
                priority: 0,
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
    if let Some((vram_slots, _)) = model.mtp_slots() {
        assert!(vram_slots >= model.config().experts.count, "every expert of the MTP block");
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
    let words: Vec<String> = prompt.iter().map(|t| format!("t{t}")).collect();
    submit_text(runtime, id, words.join(" "), max_tokens)
}

/// [`submit`] of a prompt written out: `t5 t9 ...`, and `img` for [`pictured`]'s image.
fn submit_text(
    runtime: &Runtime,
    id: u64,
    prompt: String,
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
                Event::Logprobs(_) | Event::Progress(_) | Event::Generated(_) => {}
            }
            true
        })
    };
    runtime
        .submit(
            GenerateRequest {
                model: "made".into(),
                prompt: Prompt::Text(prompt),
                sampling: Sampling::default(),
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
    rx
}

/// The greedy tokens the session writes after `prompt`.
fn session_answer(seed: u64, placement: Placement, prompt: &[u32], n: usize) -> Vec<u32> {
    session_answer_near(seed, placement, prompt, n, None)
}

/// [`session_answer`] with the K and V of the pages from `near` on in host memory.
fn session_answer_near(
    seed: u64,
    placement: Placement,
    prompt: &[u32],
    n: usize,
    near: Option<u32>,
) -> Vec<u32> {
    let devices = open_devices(placement);
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let mut model = options(placement, c.n_layer, 0, MAX_ROWS, MAX_ROWS);
    model.part.near_pages = near;
    let opts = SessionOptions { model, ram_tier: Some(64 << 20), ..SessionOptions::default() };
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

/// The error of the logits of a made-up model on the device that makes a near tie
/// ([`is_near_best`]): the bound of the residual after a layer of the shipped types against the
/// reference's in `qwen4exp_device` (4e-2, relative RMS), which lets an answer's token fall
/// behind the reference's most likely one by 0.13 of the RMS of the row's logits. (A prompt of
/// 480 rows read layer by layer began its answer with the reference's second token once, 0.07
/// behind: 3.393 against 3.496, the third at 2.721, the RMS 1.455.)
const LOGIT_ERROR: f64 = 4e-2;

/// Fails the test unless `answer`, the greedy tokens an engine wrote after `prompt` on the
/// made-up model in the file at `path`, is what the CPU reference writes up to near ties: the
/// reference reads the prompt and then the answer's own tokens, and each token of the answer
/// must be the reference's most likely one at its row or a near tie of it
/// ([`is_near_best`] at [`LOGIT_ERROR`]). The device's arithmetic is not the reference's, and a
/// made-up model has near ties that the difference flips, after which the two texts part;
/// read along the answer, the reference judges every token of it all the same. Says how far
/// the token furthest behind is.
fn assert_near_the_reference(what: &str, path: &std::path::Path, prompt: &[u32], answer: &[u32]) {
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut state = SequenceState::new(&c);
    let window = reference.window(&state, prompt, LogitRows::Last).unwrap();
    reference.commit(&mut state, &window, prompt.len()).unwrap();
    let mut logits = window.logits_of(prompt.len() - 1).unwrap().to_vec();
    let (mut worst, mut at) = (0.0f64, 0);
    for (i, &token) in answer.iter().enumerate() {
        let behind = behind_the_best(&logits, token);
        if behind > worst {
            (worst, at) = (behind, i);
        }
        let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t);
        let best = best.unwrap_or(0);
        assert!(
            is_near_best(token, &logits, LOGIT_ERROR),
            "{what}: token {i} of {answer:?} is {token}, which the reference has at {} (its most \
             likely, {best}, at {}: {behind:.3} of the RMS behind, a near tie is up to {:.3})",
            logits[token as usize],
            logits[best],
            3.0 * LOGIT_ERROR + 1e-2,
        );
        if i + 1 < answer.len() {
            let window = reference.window(&state, &[token], LogitRows::Last).unwrap();
            reference.commit(&mut state, &window, 1).unwrap();
            logits = window.logits_of(0).unwrap().to_vec();
        }
    }
    eprintln!(
        "TIE {what}: every token of {} is the reference's most likely one or a near tie; the \
         furthest behind is token {at}, {worst:.3} of the RMS",
        answer.len()
    );
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

/// [`start_with`] for the model file at `path` without a draft block or the prefix cache,
/// with steps of up to `rows` rows of which those wider than `own` run on VRAM the expert
/// slots lend, and the K and V of the pages from `near` on in page-locked host memory
/// (`blocks/far.rs`) when `near` is given.
fn start_near(
    path: &std::path::Path,
    placement: Placement,
    rows: (u32, u32),
    kv: CacheFormat,
    near: Option<u32>,
) -> (Runtime, Arc<Qwen4Exp>) {
    start_near_serving(path, placement, rows, kv, near, Serve::Measured)
}

/// [`start_near`] with the misses served as `serve` says.
fn start_near_serving(
    path: &std::path::Path,
    placement: Placement,
    (rows, own): (u32, u32),
    kv: CacheFormat,
    near: Option<u32>,
    serve: Serve,
) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(path).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let pages = 16;
    let mut opts = options(placement, n_layer, pages, MAX_ROWS, rows);
    opts.part.own_rows = own;
    opts.part.near_pages = near;
    opts.part.kv = (kv, kv);
    opts.serve = serve;
    let inner =
        Arc::new(Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap());
    let config =
        RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..RuntimeConfig::default() };
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::Off));
    let mut gpu = GpuConfig::new(&config, &*model);
    // Room for the tier of experts and the K and V beyond VRAM next to what the
    // executor needs.
    gpu.ram_limit += 512 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// A request that ends with its prompt read on lent VRAM leaves the engine idle: the loan goes
/// back then, not at the next step (`Model::idle`), so that the tiers have their slots.
#[test]
fn a_loan_goes_back_when_the_engine_idles() {
    if !enabled() {
        return;
    }
    let prompt = prompt(300, 7);
    let lending = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    for placement in placements() {
        let (runtime, model) = start_near_serving(
            &lending,
            placement,
            (LENT_ROWS, MAX_ROWS),
            CacheFormat::Q8,
            None,
            Serve::Stream,
        );
        ask(&runtime, 1, &prompt, 1);
        let begun = std::time::Instant::now();
        while model.holds_loan() == Some(true) && begun.elapsed().as_secs() < 5 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let stats = model.take_store_stats().expect("the parts are open");
        assert!(stats.scratch_loans > 0, "{placement:?}: the prompt ran on no loan");
        assert_eq!(model.holds_loan(), Some(false), "{placement:?}: the loan stayed while idle");
        runtime.shutdown();
    }
}

/// The K and V of the pages beyond the first lie in page-locked host memory, which the
/// device reads and writes over its link (`blocks/far.rs`): a prompt of three pages answers
/// with the same tokens as with every page in VRAM, in steps of the decode kernels and in
/// wide steps, in every placement. The kernels read the same bytes from another address, so
/// with every expert in a slot the answers are the same token for token, also with K and V in
/// `q6`, whose tiles are converted into the far pool as they leave the newest cells. Wide steps
/// on lent VRAM read a copy of a layer's far entries; their slots lend, so experts are fetched
/// there: with every miss uploaded, they answer token for token as with every page in VRAM.
#[test]
fn k_and_v_beyond_vram_change_no_token() {
    if !enabled() {
        return;
    }
    let prompt = prompt(300, 7);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    // Experts in BF16, whose slots lend enough for a few wide steps' scratch.
    let lending = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let reference = reference_answer_at(&lending, &prompt, 16);
    for placement in placements() {
        let cases = [
            (MAX_ROWS, CacheFormat::Q8),
            (WIDE_ROWS, CacheFormat::Q8),
            (MAX_ROWS, CacheFormat::Q6),
        ];
        for (rows, kv) in cases {
            let (mut answers, mut exact) = (Vec::new(), true);
            for near in [None, Some(1)] {
                let (runtime, model) = start_near(&gguf_path(1), placement, (rows, 0), kv, near);
                let _ = ask(&runtime, 1, &prompt, 24);
                // Where the slots of wide steps leave some expert without a slot (two
                // devices by layers), misses are served by the CPU or uploaded as the
                // costs say, and the answers agree only up to near ties.
                exact &= fill_every_slot(&model);
                let got = ask(&runtime, 2, &prompt, 24);
                if exact {
                    assert_nothing_missed(&model, "the answer with filled tiers");
                }
                runtime.shutdown();
                answers.push(got);
            }
            eprintln!(
                "FAR {placement:?}, steps of {rows} rows, K and V {kv:?}: {} of {} tokens as \
                 with every page in VRAM{}",
                leading(&answers[0], &answers[1]),
                answers[0].len(),
                if exact { "" } else { " (experts missed)" }
            );
            if exact {
                assert_eq!(answers[1], answers[0], "{placement:?}, {rows} rows, {kv:?}");
            } else {
                assert!(leading(&answers[1], &answers[0]) >= 2, "{placement:?}, {rows} rows");
            }
        }
        // Every miss of the lent steps uploaded: the GPU computes every expert. Where the CPU
        // takes a share of their misses, lent steps answer one of two first tokens here (the
        // reference ranks them 0.43 apart), every page in VRAM or not (OPEN.md: "Lent steps
        // with the CPU's share").
        let mut lent = Vec::new();
        for near in [None, Some(1)] {
            let (runtime, model) = start_near_serving(
                &lending,
                placement,
                (LENT_ROWS, MAX_ROWS),
                CacheFormat::Q8,
                near,
                Serve::Stream,
            );
            assert_eq!(runtime_rows(&model), (LENT_ROWS, MAX_ROWS));
            lent.push(ask(&runtime, 1, &prompt, 16));
            let stats = model.take_store_stats().expect("the parts are open");
            runtime.shutdown();
            assert!(stats.scratch_loans > 0, "{placement:?}: the prompt ran on no loan");
        }
        eprintln!(
            "FAR {placement:?}, lent steps of {LENT_ROWS} rows: {} and {} tokens as the \
             reference's (every page in VRAM, K and V beyond the first page in RAM), {} alike",
            leading(&lent[0], &reference),
            leading(&lent[1], &reference),
            leading(&lent[0], &lent[1])
        );
        assert!(leading(&lent[0], &reference) >= 2, "{placement:?}: {:?}", lent[0]);
        assert_eq!(lent[1], lent[0], "{placement:?}: lent steps with K and V beyond VRAM");
        // The session with its pages beyond the first in RAM says what the session says.
        let want = session_answer(1, placement, &prompt, 24);
        assert_eq!(session_answer_near(1, placement, &prompt, 24, Some(1)), want, "{placement:?}");
    }
}

/// Fills the tiers as [`warm`] does; whether every expert has a slot in VRAM.
fn fill_every_slot(model: &Qwen4Exp) -> bool {
    let n_experts = model.config().experts.count as u32;
    model
        .fill(&mut |_| (0..n_experts).collect())
        .expect("the parts are open")
        .expect("the tiers fill");
    let (vram_slots, _, _) = model.expert_slots().expect("the parts are open");
    model.take_store_stats();
    vram_slots >= model.config().n_layer * model.config().experts.count
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

/// A step of more rows than one launch of the block scores holds (256) scores and selects them
/// a few hundred at a time: a prompt of 480 tokens read in one step answers as the CPU
/// reference does, in every placement (later tokens follow near ties, as in the test above).
#[test]
fn a_wide_step_scores_its_rows_a_few_hundred_at_a_time() {
    if !enabled() {
        return;
    }
    let prompt = prompt(480, 9);
    let reference = reference_answer(1, &prompt, 16);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    for placement in placements() {
        let (runtime, model) = start_rows(1, placement, 16, RuntimeConfig::default(), 512);
        let got = ask(&runtime, 1, &prompt, 16);
        let wide = model.wide_stats().expect("the parts are open");
        runtime.shutdown();
        eprintln!(
            "SCORES {placement:?}: {} tokens as the reference's before the first difference",
            leading(&got, &reference)
        );
        assert!(wide.layers > 0, "{placement:?}: the prompt ran in no wide step");
        assert!(leading(&got, &reference) >= 2, "{placement:?}: {got:?} against {reference:?}");
    }
}

/// A prompt read alone runs in wide steps of more rows than the parts' own scratch holds,
/// on VRAM the expert slots lend for it (`expert_store/scratch.rs`), in every placement: it
/// answers as the same prompt read in steps of the own scratch does (both start as the CPU
/// reference's answer; later tokens of a made-up model follow near ties, which the order of
/// the experts' sums in a wide step flips), and the slots lend for the prompt and take the
/// VRAM back after it.
#[test]
fn a_prompt_read_alone_runs_on_vram_the_expert_slots_lend() {
    if !enabled() {
        return;
    }
    let prompt = prompt(200, 5);
    // Experts in BF16, whose slots lend enough for a few wide steps' scratch.
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let reference = reference_answer_at(&path, &prompt, 16);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    for placement in placements() {
        let (runtime, _) = start_at(&path, placement, 16, RuntimeConfig::default(), WIDE_ROWS);
        let own = ask(&runtime, 1, &prompt, 16);
        runtime.shutdown();
        // Steps of 32 rows on the loan, of eight in the parts' own memory.
        let (runtime, model) = start_lending(&path, placement, 16, LENT_ROWS, MAX_ROWS);
        assert_eq!(runtime_rows(&model), (LENT_ROWS, MAX_ROWS));
        let lent = ask(&runtime, 1, &prompt, 16);
        // Asked again, the same conversation reads its prompt from the cache and
        // the steps run in the parts' own memory.
        let again = ask(&runtime, 2, &prompt, 16);
        let stats = model.take_store_stats().expect("the parts are open");
        runtime.shutdown();
        eprintln!(
            "LENT {placement:?}: {} tokens as the reference's before the first difference (own \
             scratch: {}), {} as the own scratch's; {} loans, {} experts moved out of the lent \
             slots",
            leading(&lent, &reference),
            leading(&own, &reference),
            leading(&lent, &own),
            stats.scratch_loans,
            stats.scratch_moves
        );
        // Each device lends for the first prompt, and again for what the second
        // reads past the cache, if that is more than the own scratch holds.
        let devices = if placement == Placement::One { 1 } else { 2 };
        assert!(
            (devices..=2 * devices).contains(&stats.scratch_loans),
            "{placement:?}: {} loans",
            stats.scratch_loans
        );
        assert!(leading(&lent, &reference) >= 2, "{placement:?}: {lent:?} against {reference:?}");
        assert!(leading(&again, &reference) >= 2, "{placement:?}: {again:?} after the loan");
    }
}

/// A prompt read alone in a step wider than the parts' own scratch runs layer by layer
/// (`device/runtime/pass.rs`): every micro-batch of the parts' own rows through a layer, then
/// the layer's experts once for all rows, the rows kept on VRAM the expert slots lend. It
/// answers as the CPU reference does (the first tokens; later ones of a made-up model follow
/// near ties, which the order of the experts' sums flips) and as the same prompt read in steps
/// of the own scratch, with micro-batches in one chunk and in several.
#[test]
fn a_prompt_read_layer_by_layer_gives_the_reference_answer() {
    if !enabled() {
        return;
    }
    let prompt = prompt(400, 5);
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let reference = reference_answer_at(&path, &prompt, 16);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    for placement in placements() {
        for own in [32, 16] {
            let (runtime, _) = start_passing(&path, placement, 16, 0, own);
            let steps = ask(&runtime, 1, &prompt, 16);
            runtime.shutdown();
            let (runtime, model) = start_passing(&path, placement, 16, 512, own);
            assert_eq!(runtime_rows(&model), (512, own));
            let passed = ask(&runtime, 1, &prompt, 16);
            // Asked again, the prompt comes from the cache: its last rows run as a step.
            let again = ask(&runtime, 2, &prompt, 16);
            let stats = model.take_store_stats().expect("the parts are open");
            let wide = model.wide_stats().expect("the parts are open");
            runtime.shutdown();
            eprintln!(
                "PASS {placement:?} own {own}: {} tokens as the reference's before the first \
                 difference (steps of {own} rows: {}), {} as those steps'; {} loans; {} experts \
                 missed, {} of them on the CPU",
                leading(&passed, &reference),
                leading(&steps, &reference),
                leading(&passed, &steps),
                stats.scratch_loans,
                wide.missing,
                wide.cpu_experts,
            );
            assert!(stats.scratch_loans >= 1, "{placement:?}: no pass lent VRAM");
            assert!(
                leading(&passed, &reference) >= 2,
                "{placement:?}: {passed:?} against {reference:?}"
            );
            assert!(leading(&again, &reference) >= 2, "{placement:?}: {again:?} after the pass");
        }
    }
}

/// A prompt read layer by layer (a pass) with the K and V of the pages beyond the first in
/// page-locked host memory (`blocks/far.rs`): every micro-batch's attention reads a copy of
/// the layer's far slots in the pass's loan, which it brings up to date behind its own append
/// (`Part::pass_stage`). The answer starts as the CPU reference's, as the same pass's with
/// every page in VRAM does, in micro-batches of one chunk and of several, in every placement;
/// asked again, the prompt comes from the cache and its last rows, a step of their own, read
/// the far pool in place.
#[test]
fn a_prompt_read_layer_by_layer_reads_k_and_v_beyond_vram_from_a_copy() {
    if !enabled() {
        return;
    }
    let prompt = prompt(400, 5);
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let reference = reference_answer_at(&path, &prompt, 16);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    for placement in placements() {
        for own in [32, 16] {
            let mut answers = Vec::new();
            for near in [None, Some(1)] {
                let (runtime, model) = start_passing_near(&path, placement, 512, own, near);
                assert_eq!(runtime_rows(&model), (512, own));
                let passed = ask(&runtime, 1, &prompt, 16);
                let again = ask(&runtime, 2, &prompt, 16);
                let stats = model.take_store_stats().expect("the parts are open");
                runtime.shutdown();
                assert!(stats.scratch_loans >= 1, "{placement:?}: no pass lent VRAM");
                answers.push((passed, again));
            }
            let ((vram, _), (far, again)) = (&answers[0], &answers[1]);
            eprintln!(
                "FAR PASS {placement:?} own {own}: {} tokens as the reference's (every page in \
                 VRAM: {}), {} as with every page in VRAM; again from the cache {}",
                leading(far, &reference),
                leading(vram, &reference),
                leading(far, vram),
                leading(again, &reference),
            );
            assert!(leading(far, &reference) >= 2, "{placement:?}: {far:?} against {reference:?}");
            assert!(leading(again, &reference) >= 2, "{placement:?}: {again:?} from the cache");
        }
    }
}

/// A request that starts on a page beyond VRAM: the prompt cache holds the pages of a prompt
/// of three pages, so the next request's sequence begins on the pool's lowest free page, which
/// is far when only the first page is near; and one that shares the first 50 cells of that
/// prompt, whose first page the cache copies into a far page. Their answers are the ones they
/// give with every page in VRAM, in every placement.
#[test]
fn a_sequence_that_starts_beyond_vram_answers_as_in_vram() {
    if !enabled() {
        return;
    }
    let (first, next) = (prompt(300, 7), prompt(40, 11));
    let shared: Vec<u32> = first[..50].iter().copied().chain(prompt(30, 13)).collect();
    for placement in placements() {
        let mut answers = Vec::new();
        for near in [None, Some(1)] {
            let (runtime, model) = start_passing_near(&gguf_path(1), placement, 0, MAX_ROWS, near);
            let _ = ask(&runtime, 1, &first, 8);
            let exact = fill_every_slot(&model);
            let mut got = ask(&runtime, 2, &next, 16);
            got.extend(ask(&runtime, 3, &shared, 16));
            runtime.shutdown();
            answers.push((got, exact));
        }
        let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
        eprintln!(
            "FAR START {placement:?}: {} of {} tokens as with every page in VRAM",
            leading(&answers[1].0, &answers[0].0),
            answers[0].0.len()
        );
        if answers.iter().all(|(_, exact)| *exact) {
            assert_eq!(answers[1].0, answers[0].0, "{placement:?}");
        } else {
            assert!(leading(&answers[1].0, &answers[0].0) >= 2, "{placement:?}");
        }
    }
}

/// [`start_passing`] on 16 pages with the K and V of the pages from `near` on in host memory.
fn start_passing_near(
    path: &std::path::Path,
    placement: Placement,
    pass_rows: u32,
    own: u32,
    near: Option<u32>,
) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(path).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let pages = 16;
    let mut opts = options(placement, n_layer, pages, MAX_ROWS, own);
    opts.pass_rows = pass_rows;
    opts.serve = Serve::Measured;
    opts.part.near_pages = near;
    let inner =
        Arc::new(Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap());
    let config =
        RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..RuntimeConfig::default() };
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    // Room for the tier of experts and the K and V beyond VRAM next to what the
    // executor needs.
    gpu.ram_limit += 512 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// The image of [`pictured`]: `IMAGE.0` rows across and `IMAGE.1` down, its cells the
/// n-gram block's image token.
const IMAGE: (u32, u32) = (8, 5);
const PAD: u32 = 9;

/// The image the word `img` stands for.
fn image() -> Arc<ImageInput> {
    let (nx, ny) = IMAGE;
    Arc::new(ImageInput {
        name: [7; 32],
        width: nx * 32,
        height: ny * 32,
        pixels: Vec::new(),
        nx,
        ny,
    })
}

/// The made-up model with images: the word `img` of a prompt is an image of [`IMAGE`]'s
/// rows, which the tower of its first device part gives from `rows` (made up, as a
/// projector would compute them).
fn pictured(inner: &Arc<Qwen4Exp>, width: u32, rows: Arc<Vec<f32>>) -> Wrapped {
    let part = inner.clone();
    made(inner, Cache::On)
        .encode(|prompt| {
            let Prompt::Text(text) = prompt else {
                return Err(EngineError::Unsupported(
                    "the made-up model has no chat template".into(),
                ));
            };
            let mut out = EncodedPrompt::default();
            for word in text.split_whitespace() {
                if word == "img" {
                    let first = out.tokens.len() as u32;
                    out.images.push(ImageSpan { first, image: image() });
                    out.tokens.extend(std::iter::repeat_n(PAD, (IMAGE.0 * IMAGE.1) as usize));
                } else {
                    out.tokens.extend(words(&Prompt::Text(word.into()))?);
                }
            }
            Ok(out)
        })
        .vision(Vision { width, max_rows: 64, device: 0, units: 1 })
        .open(move |cx| Ok(Box::new(Towered { part: part.open(cx)?, rows: rows.clone() })))
}

/// A device part of [`pictured`]: the model's, with a tower that gives the made-up rows.
struct Towered {
    part: Box<dyn DeviceModel>,
    rows: Arc<Vec<f32>>,
}

impl DeviceModel for Towered {
    fn logits_per_row(&self) -> u32 {
        self.part.logits_per_row()
    }
    fn enqueue(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        self.part.enqueue(step)
    }
    fn wait(&mut self, end: &ares_cuda::Event) -> Result<(), ModelError> {
        self.part.wait(end)
    }
    fn picked(&mut self, picks: &[Pick]) {
        self.part.picked(picks)
    }
    fn logit_tokens(&self) -> ares_cuda::DevicePtr {
        self.part.logit_tokens()
    }
    fn model_drafts(&self) -> Option<ModelDrafts> {
        self.part.model_drafts()
    }
    fn transfer(
        &mut self,
        cx: &TransferContext<'_>,
        item: &mut DeviceTransfer,
    ) -> Result<(), ModelError> {
        self.part.transfer(cx, item)
    }
    fn tower(&mut self, _stream: &ares_cuda::Stream, job: &mut TowerJob) -> Result<(), ModelError> {
        job.rows = self.rows.to_vec();
        job.ms = 0.1;
        Ok(())
    }
    fn model_draft_ms(&mut self) -> f64 {
        self.part.model_draft_ms()
    }
    fn model_draft_probs(&mut self, out: &mut Vec<f32>) {
        self.part.model_draft_probs(out)
    }
    fn rows_not_exact(&mut self, out: &mut Vec<u32>) {
        self.part.rows_not_exact(out)
    }
    fn paused(&self) -> bool {
        self.part.paused()
    }
}

/// The greedy tokens of the CPU reference after `prompt`, whose rows `inputs.rows` are an
/// image's at the rotary positions `spans` gives every cell.
fn reference_answer_with(
    path: &std::path::Path,
    prompt: &[u32],
    rows: &[(usize, Vec<f32>)],
    spans: Option<&Spans>,
    n: usize,
) -> Vec<u32> {
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut state = SequenceState::new(&c);
    let arg_max = |row: &[f32]| {
        row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
    };
    let rotary = |cells: std::ops::Range<usize>| {
        spans.map(|s| cells.map(|cell| s.rotary(cell as u32).map(|p| p as usize)).collect())
    };
    let inputs = Inputs { rows: rows.to_vec(), rotary: rotary(0..prompt.len()) };
    let mut window = reference.window_with(&state, prompt, &inputs, LogitRows::All).unwrap();
    reference.commit(&mut state, &window, prompt.len()).unwrap();
    let mut answer = vec![arg_max(window.logits_of(prompt.len() - 1).unwrap())];
    while answer.len() < n {
        let token = [answer[answer.len() - 1]];
        let cell = state.position();
        let one = Inputs { rows: Vec::new(), rotary: rotary(cell..cell + 1) };
        window = reference.window_with(&state, &token, &one, LogitRows::Last).unwrap();
        reference.commit(&mut state, &window, 1).unwrap();
        answer.push(arg_max(window.logits_of(0).unwrap()));
    }
    answer
}

/// A prompt with an image read layer by layer: the micro-batches that hold the image's cells
/// take the rows the runtime staged (the made-up tower's) over their gathered rows, and every
/// row its rotary position (the image's rows `(p, p + j / nx, p + j % nx)`, the text after it
/// `max(nx, ny)` on). It answers as the CPU reference given the same rows and positions does,
/// and as the same prompt read in steps of the own scratch; asked again, it comes from the
/// cache with the image's cells replayed from the store's rows.
#[test]
fn a_prompt_with_an_image_read_layer_by_layer_gives_the_reference_answer() {
    if !enabled() {
        return;
    }
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let (nx, ny) = IMAGE;
    let (first, image_rows) = (150usize, (nx * ny) as usize);
    let text = prompt(360, 7);
    let mut tokens = text[..first].to_vec();
    tokens.extend(std::iter::repeat_n(PAD, image_rows));
    tokens.extend_from_slice(&text[first..]);
    let words: Vec<String> = text.iter().map(|t| format!("t{t}")).collect();
    let words = format!("{} img {}", words[..first].join(" "), words[first..].join(" "));
    // The image's rows: two token embeddings mixed, so that their size is a token's.
    let gguf = Gguf::open(&path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let embedding = |t: u32| {
        let mut row = vec![0.0f32; c.n_embd];
        w.token_embd.decode_row(t as usize, &mut row);
        row
    };
    let rows: Vec<(usize, Vec<f32>)> = (0..image_rows as u32)
        .map(|j| {
            let (a, b) = (embedding(10 + 3 * j % 80), embedding(12 + 7 * j % 78));
            (first + j as usize, a.iter().zip(&b).map(|(x, y)| 0.7 * x + 0.9 * y).collect())
        })
        .collect();
    let tower_rows = Arc::new(rows.iter().flat_map(|(_, r)| r.iter().copied()).collect::<Vec<_>>());
    let width = c.n_embd as u32;
    drop(w);
    let spans = Spans::new(vec![ImageSpan { first: first as u32, image: image() }]);
    let reference = reference_answer_with(&path, &tokens, &rows, Some(&spans), 16);
    // What the test tells apart: the pad token's embedding at the image's cells, and the
    // image's rows at their cells.
    let plain = reference_answer_with(&path, &tokens, &[], None, 16);
    let at_cells = reference_answer_with(&path, &tokens, &rows, None, 16);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    eprintln!(
        "IMAGE reference {reference:?}; without the rows {} tokens alike, without the \
         positions {}",
        leading(&plain, &reference),
        leading(&at_cells, &reference)
    );
    assert!(leading(&plain, &reference) < 2, "the image's rows should matter");
    let start = |placement: Placement, pass_rows: u32, own: u32| {
        let devices = open_devices(placement);
        let mut opts = options(placement, c.n_layer, 16, MAX_ROWS, own);
        opts.pass_rows = pass_rows;
        opts.serve = Serve::Measured;
        let gguf = Gguf::open(&path).unwrap();
        let inner = Arc::new(
            Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap(),
        );
        let config = RuntimeConfig {
            pages: 16,
            cells_per_page: 128,
            max_slots: SLOTS,
            ..RuntimeConfig::default()
        };
        let model: Arc<dyn Model> = Arc::new(pictured(&inner, width, tower_rows.clone()));
        let mut gpu = GpuConfig::new(&config, &*model);
        gpu.ram_limit += 256 << 20;
        gpu.vram_headroom = 512 << 20;
        (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
    };
    let ask_words = |runtime: &Runtime, id: u64| {
        let rx = submit_text(runtime, id, words.clone(), 16);
        rx.recv_timeout(Duration::from_secs(300)).expect("an answer in time").unwrap()
    };
    for placement in placements() {
        for own in [32, 16] {
            let (runtime, _) = start(placement, 0, own);
            let steps = ask_words(&runtime, 1);
            runtime.shutdown();
            let (runtime, model) = start(placement, 512, own);
            let passed = ask_words(&runtime, 1);
            let again = ask_words(&runtime, 2);
            let stats = model.take_store_stats().expect("the parts are open");
            runtime.shutdown();
            eprintln!(
                "IMAGE PASS {placement:?} own {own}: {} tokens as the reference's (steps of \
                 {own} rows: {}), {} as those steps'; again from the cache {}; {} loans",
                leading(&passed, &reference),
                leading(&steps, &reference),
                leading(&passed, &steps),
                leading(&again, &reference),
                stats.scratch_loans,
            );
            assert!(stats.scratch_loans >= 1, "{placement:?}: no pass lent VRAM");
            assert!(
                leading(&passed, &reference) >= 2,
                "{placement:?}: {passed:?} against {reference:?}"
            );
            assert!(leading(&steps, &reference) >= 2, "{placement:?}: {steps:?} in steps");
            assert!(leading(&again, &reference) >= 2, "{placement:?}: {again:?} from the cache");
        }
    }
}

/// A request that comes while a long prompt is read layer by layer gets its first token
/// before that prompt's: the pass pauses between two micro-batches when the engine is asked
/// something, the new request's prompt runs, and the pass goes on where it stopped. Both
/// answer as they do alone.
#[test]
fn a_request_cuts_in_between_the_pieces_of_a_long_prompt() {
    if !enabled() {
        return;
    }
    let long = prompt(480, 7);
    let short = prompt(12, 8);
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let short_reference = reference_answer_at(&path, &short, 8);
    let long_reference = reference_answer_at(&path, &long, 8);
    for placement in placements() {
        let (runtime, _) = start_passing(&path, placement, 16, 512, 16);
        let long_alone = ask(&runtime, 1, &long, 8);
        let short_alone = ask(&runtime, 2, &short, 8);
        runtime.shutdown();
        // The same prompt in steps of the own scratch, which in this placement may part
        // from the reference at a near tie.
        let (runtime, _) = start_passing(&path, placement, 16, 0, 16);
        let long_steps = ask(&runtime, 1, &long, 8);
        runtime.shutdown();
        let (runtime, _) = start_passing(&path, placement, 16, 512, 16);
        let begun = std::time::Instant::now();
        let (long_rx, long_first) = submit_timed(&runtime, 3, &long, 8);
        std::thread::sleep(Duration::from_millis(5));
        let asked = begun.elapsed();
        let (short_rx, short_first) = submit_timed(&runtime, 4, &short, 8);
        let short_got = short_rx.recv_timeout(Duration::from_secs(300)).unwrap().unwrap();
        let long_got = long_rx.recv_timeout(Duration::from_secs(300)).unwrap().unwrap();
        let paused = runtime.snapshot().expect("the runtime is running").counters.steps_paused;
        runtime.shutdown();
        let (short_at, long_at) =
            (short_first.lock().unwrap().unwrap(), long_first.lock().unwrap().unwrap());
        eprintln!(
            "CUT {placement:?}: short asked after {asked:?}, its first token after {:?}; the long \
             prompt's after {:?}; {paused} steps paused",
            short_at.duration_since(begun),
            long_at.duration_since(begun),
        );
        eprintln!(
            "CUT {placement:?}: short {short_got:?} (alone {short_alone:?}, reference \
             {short_reference:?}); long {long_got:?} (alone {long_alone:?}, reference \
             {long_reference:?}, in steps {long_steps:?})"
        );
        assert!(paused >= 1, "{placement:?}: the long prompt's pass never paused");
        assert!(short_at < long_at, "{placement:?}: the short request waited for the long one");
        assert_near_the_reference(&format!("{placement:?}, short"), &path, &short, &short_got);
        assert_near_the_reference(&format!("{placement:?}, long"), &path, &long, &long_got);
    }
}

/// When a request's first text arrived, once it has.
type FirstText = Arc<Mutex<Option<std::time::Instant>>>;

/// [`submit`], which also notes when the request's first text arrived.
fn submit_timed(
    runtime: &Runtime,
    id: u64,
    prompt: &[u32],
    max_tokens: u32,
) -> (mpsc::Receiver<Result<Vec<u32>, String>>, FirstText) {
    let (tx, rx) = mpsc::channel();
    let first = Arc::new(Mutex::new(None));
    let text = Arc::new(Mutex::new(String::new()));
    let sink = {
        let (text, first) = (text.clone(), first.clone());
        EventSink::new(move |event| {
            match event {
                Event::Text(piece) => {
                    first.lock().unwrap().get_or_insert_with(std::time::Instant::now);
                    text.lock().unwrap().push_str(&piece);
                }
                Event::Done { .. } => {
                    let _ = tx.send(Ok(tokens_of(&text.lock().unwrap())));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
                Event::Logprobs(_) | Event::Progress(_) | Event::Generated(_) => {}
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
                priority: 0,
            },
            sink,
        )
        .unwrap();
    (rx, first)
}

/// A prompt read layer by layer with few enough rows has the CPU compute the experts it misses
/// that are in RAM (`Serve::Cpu`: every one it can): the micro-batches' mixed input reaches the
/// host in the layout of all the pass's rows, and the CPU's sums go into their chunks' rows.
/// The prompt's token is the reference's in every placement.
#[test]
fn a_prompt_read_layer_by_layer_has_the_cpu_compute_its_misses() {
    if !enabled() {
        return;
    }
    let warm_up = prompt(400, 9);
    let prompt = prompt(400, 5);
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let reference = reference_answer_at(&path, &prompt, 16);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    for placement in placements() {
        // The same in steps of the own rows, whose misses the CPU takes as well.
        let (runtime, _) = start_serving(&path, placement, 16, 0, 32, Serve::Cpu, Some(16 << 20));
        let _ = ask(&runtime, 1, &warm_up, 4);
        let steps = ask(&runtime, 2, &prompt, 16);
        runtime.shutdown();
        let (runtime, model) =
            start_serving(&path, placement, 16, 512, 32, Serve::Cpu, Some(16 << 20));
        // A first prompt brings experts into RAM, whose misses the CPU takes.
        let _ = ask(&runtime, 1, &warm_up, 4);
        let passed = ask(&runtime, 2, &prompt, 16);
        let wide = model.wide_stats().expect("the parts are open");
        runtime.shutdown();
        eprintln!(
            "PASS-CPU {placement:?}: {} tokens as the reference's (in steps: {}), {} as the \
             steps'; {} experts missed, {} on the CPU for {} pairs; {passed:?} {steps:?} \
             {reference:?}",
            leading(&passed, &reference),
            leading(&steps, &reference),
            leading(&passed, &steps),
            wide.missing,
            wide.cpu_experts,
            wide.cpu_pairs,
        );
        assert!(wide.cpu_experts > 0, "{placement:?}: the CPU computed nothing");
        // Which misses the CPU takes depends on the times it measures, and its 8-bit rows
        // flip a made-up model's near ties after the prompt from run to run (in steps too):
        // the prompt's own token is the reference's.
        assert!(leading(&passed, &reference) >= 1, "{placement:?}: {passed:?}");
    }
}

/// [`start_at`] whose prompts read alone run layer by layer in steps of up to `pass_rows`
/// rows (0: in steps of the own scratch), in micro-batches of `own` rows.
fn start_passing(
    path: &std::path::Path,
    placement: Placement,
    pages: u32,
    pass_rows: u32,
    own: u32,
) -> (Runtime, Arc<Qwen4Exp>) {
    start_serving(path, placement, pages, pass_rows, own, Serve::Measured, None)
}

/// A prompt read layer by layer saves the state where checkpoints of the prefix cache are due
/// inside it, layer by layer as it goes past them, and is not cut there: a later prompt that
/// shares the first part resumes from such a checkpoint and answers as the reference does, in
/// every placement.
#[test]
fn a_prompt_read_layer_by_layer_leaves_checkpoints_inside_it() {
    if !enabled() {
        return;
    }
    let first = prompt(400, 5);
    let mut later = first[..300].to_vec();
    later.extend(prompt(60, 11));
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let reference = reference_answer_at(&path, &later, 16);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let config = RuntimeConfig {
        cache_ram: 256 << 20,
        checkpoint_prompt_interval: 128,
        ..RuntimeConfig::default()
    };
    for placement in placements() {
        let (runtime, _) = start_serving_with(
            &path,
            placement,
            16,
            512,
            32,
            Serve::Measured,
            None,
            config.clone(),
        );
        let _ = ask(&runtime, 1, &first, 4);
        let saved = runtime.snapshot().expect("the runtime runs").counters;
        let again = ask(&runtime, 2, &later, 16);
        let counters = runtime.snapshot().expect("the runtime runs").counters;
        runtime.shutdown();
        eprintln!(
            "INSIDE {placement:?}: {} checkpoints after the first prompt ({} prompt rows), the \
             later one {} restored, {} rows computed in all; {} tokens as the reference's",
            saved.checkpoints_saved,
            saved.prefill_rows,
            counters.checkpoints_restored,
            counters.prefill_rows,
            leading(&again, &reference),
        );
        assert!(saved.checkpoints_saved >= 2, "{placement:?}: no checkpoint inside the pass");
        assert!(counters.checkpoints_restored >= 1, "{placement:?}: nothing was restored");
        assert!(leading(&again, &reference) >= 2, "{placement:?}: {again:?} against {reference:?}");
    }
}

/// An agent's next turn right after a short one, with one checkpoint place: the short turn's
/// end state is saved with the next record, the next turn restores it in that record, and its
/// rows are read layer by layer with checkpoints due inside them, whose places are taken
/// before the record is written and give the end state up. The save still runs before the
/// restore, and the turn answers as the reference does, in every placement. (Without it the
/// restore read a place nothing was saved in, and the step failed: "the rows of slot 0 start
/// at cell 53 and its sequence is at cell 0".)
#[test]
fn a_turn_read_layer_by_layer_from_an_end_state_saved_with_its_record_answers_as_the_reference() {
    if !enabled() {
        return;
    }
    let first = prompt(50, 5);
    let path = gguf_path_of(1, GgmlType::BF16, GgmlType::Q4_0);
    let leading = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    for placement in placements() {
        // The bytes of one checkpoint, for a budget that holds one.
        let (probe, inner) = start_passing(&path, placement, 16, 512, 32);
        let bytes = inner.prefix_cache().expect("the model takes part in the cache").bytes();
        probe.shutdown();
        let config = RuntimeConfig {
            cache_ram: bytes + bytes / 2,
            checkpoint_prompt_interval: 128,
            ..RuntimeConfig::default()
        };
        let (runtime, _) =
            start_serving_with(&path, placement, 16, 512, 32, Serve::Measured, None, config);
        // Less than a page with its answer: the end state is the one checkpoint.
        let start = runtime.snapshot().expect("the runtime runs").counters;
        let said = ask(&runtime, 1, &first, 4);
        let saved = runtime.snapshot().expect("the runtime runs").counters;
        let mut next = first.clone();
        next.extend(&said);
        next.extend(prompt(300, 11));
        let again = ask(&runtime, 2, &next, 16);
        let counters = runtime.snapshot().expect("the runtime runs").counters;
        runtime.shutdown();
        let reference = reference_answer_at(&path, &next, 16);
        let saves = saved.checkpoints_saved - start.checkpoints_saved;
        assert_eq!(saves, 1, "{placement:?}: the end state is saved");
        let restores = counters.checkpoints_restored - saved.checkpoints_restored;
        assert_eq!(restores, 1, "{placement:?}: the end state is restored");
        let replayed = counters.replay_rows - saved.replay_rows;
        assert_eq!(replayed, 0, "{placement:?}: nothing is replayed");
        assert!(leading(&again, &reference) >= 2, "{placement:?}: {again:?} against {reference:?}");
    }
}

/// [`start_passing`] whose misses are served as `serve` says, with `slot_bytes` bytes of
/// expert slots in VRAM per device when given.
fn start_serving(
    path: &std::path::Path,
    placement: Placement,
    pages: u32,
    pass_rows: u32,
    own: u32,
    serve: Serve,
    slot_bytes: Option<u64>,
) -> (Runtime, Arc<Qwen4Exp>) {
    let config = RuntimeConfig::default();
    start_serving_with(path, placement, pages, pass_rows, own, serve, slot_bytes, config)
}

/// [`start_serving`] with the runtime's settings `config` (its pages, cells per page and
/// slots are the test's).
#[allow(clippy::too_many_arguments)]
fn start_serving_with(
    path: &std::path::Path,
    placement: Placement,
    pages: u32,
    pass_rows: u32,
    own: u32,
    serve: Serve,
    slot_bytes: Option<u64>,
    config: RuntimeConfig,
) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(path).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let mut opts = options(placement, n_layer, pages, MAX_ROWS, own);
    opts.pass_rows = pass_rows;
    opts.serve = serve;
    if slot_bytes.is_some() {
        opts.slot_bytes = slot_bytes;
        opts.reserve = None;
    }
    let inner =
        Arc::new(Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap());
    let config = RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..config };
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// The rows of a step on VRAM the made-up model's expert slots lend.
const LENT_ROWS: u32 = 32;

/// The rows of a step the model takes, and those in its parts' own memory.
fn runtime_rows(model: &Qwen4Exp) -> (u32, u32) {
    (model.max_rows(), model.own_rows())
}

/// [`start_at`] with steps of up to `rows` rows, of which those wider than `own` run on
/// VRAM the expert slots lend.
fn start_lending(
    path: &std::path::Path,
    placement: Placement,
    pages: u32,
    rows: u32,
    own: u32,
) -> (Runtime, Arc<Qwen4Exp>) {
    let devices = open_devices(placement);
    let gguf = Gguf::open(path).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let mut opts = options(placement, n_layer, pages, MAX_ROWS, rows);
    opts.part.own_rows = own;
    let inner =
        Arc::new(Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap());
    let config =
        RuntimeConfig { pages, cells_per_page: 128, max_slots: SLOTS, ..RuntimeConfig::default() };
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
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
                    priority: 0,
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
/// reference of the model and of its MTP block see them: the residual
/// after the last layer of every cell, and the block's cells that follow.
struct Followed {
    tokens: Vec<u32>,
    trunk: Vec<f32>,
    wide: usize,
}

/// The chains the MTP block of the made-up model `seed` made for a request
/// that read `prompt` and wrote `answer`, against the CPU reference's chains
/// from the same cells: per link, how many were compared (a link is compared
/// when the links before it agreed) and how many agreed.
fn chains_against_the_reference(
    seed: u64,
    chains: &[ares_model::qwen4exp::device::MtpDrafted],
    prompt: &[u32],
    answer: &[u32],
) -> ([usize; 8], [usize; 8]) {
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let file = Gguf::open(draft_path(seed)).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let head = MtpWeights::load(&file, &c).unwrap();
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
            &MtpState::new(),
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
        let mut state = MtpState::new();
        reference.mtp_commit(&mut state, &follow, newest - 1).unwrap();
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

/// The model drafts with its own MTP block in every placement, and the
/// block on the device drafts what the CPU reference's chain drafts from the
/// same cells: its input from the model's residual and the paired token, its
/// keys and values of the cells the model has passed, its experts through
/// the store, its own final mixer and the model's head, link after link.
#[test]
fn the_mtp_block_drafts_what_the_reference_drafts_in_every_placement() {
    if !enabled() {
        return;
    }
    let prompt = prompt(21, 3);
    for placement in placements() {
        let (runtime, model) = start_drafting_unpriced(1, placement, 16);
        warm(&model);
        model.take_mtp_chains();
        let answer = ask(&runtime, 1, &prompt, 40);
        assert_nothing_missed(&model, "an answer with drafts and filled tiers");
        let chains = model.take_mtp_chains();
        let counters = runtime.snapshot().expect("the runtime is running").counters;
        let report = model.mtp_report().expect("the model has an MTP block");
        let times = model.mtp_times();
        runtime.shutdown();
        let (compared, agreed) = chains_against_the_reference(1, &chains, &prompt, &answer);
        eprintln!(
            "RES {placement:?}: {} chains of {} links in all; against the reference per link \
             {agreed:?} of {compared:?}; draft rows {}, accepted {}, rejected {}; the block on \
             part {}: weights {} B, pools {} B, buffers {} B; {:.2} ms a link",
            chains.len(),
            counters.model_draft_links,
            counters.draft_rows,
            counters.drafts_accepted,
            counters.drafts_rejected,
            report.index,
            report.weights,
            report.pools,
            report.buffers,
            times.chain_ms / times.links.max(1) as f64,
        );
        assert!(counters.model_draft_links > 0, "{placement:?}: the block drafted nothing");
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

/// The MTP block follows a prompt read layer by layer micro-batch by micro-batch, after the
/// pass (its last micro-batch at the next step's start, from the residual the pass copies to
/// the own scratch): its chains after the prompt draft what the reference's draft from the same
/// cells, in every placement.
#[test]
fn the_mtp_block_follows_a_prompt_read_layer_by_layer() {
    if !enabled() {
        return;
    }
    let prompt = prompt(300, 3);
    for placement in placements() {
        let devices = open_devices(placement);
        let gguf = Gguf::open(gguf_path(1)).unwrap();
        let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
        let mut opts = options(placement, n_layer, 16, MAX_ROWS, 16);
        opts.pass_rows = 512;
        // Every miss uploaded: which ones the CPU would take depends on the times it measures,
        // and its 8-bit rows move a made-up model's near ties.
        opts.serve = Serve::Stream;
        let inner = Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS)
            .unwrap()
            .with_mtp(Gguf::open(draft_path(1)).unwrap())
            .expect("the MTP block fits");
        let model = Arc::new(inner);
        let config = RuntimeConfig {
            pages: 16,
            cells_per_page: 128,
            max_slots: SLOTS,
            window_rows: 8,
            warm_up: false,
            ..RuntimeConfig::default()
        };
        let wrapped: Arc<dyn Model> = Arc::new(made(&model, Cache::On));
        let mut gpu = GpuConfig::new(&config, &*wrapped);
        gpu.ram_limit += 256 << 20;
        gpu.vram_headroom = 512 << 20;
        let runtime =
            Runtime::start_gpu(wrapped, devices, config, gpu).expect("the runtime starts");
        model.take_mtp_chains();
        let answer = ask(&runtime, 1, &prompt, 40);
        let chains = model.take_mtp_chains();
        let counters = runtime.snapshot().expect("the runtime is running").counters;
        let loans = model.take_store_stats().expect("the parts are open").scratch_loans;
        runtime.shutdown();
        let (compared, agreed) = chains_against_the_reference(1, &chains, &prompt, &answer);
        eprintln!(
            "MTP-PASS {placement:?}: a pass of up to {} rows, {loans} loans; {} chains; against \
             the reference per link {agreed:?} of {compared:?}; draft rows {}, accepted {}",
            model.max_rows(),
            chains.len(),
            counters.draft_rows,
            counters.drafts_accepted,
        );
        assert!(model.max_rows() > 16 && loans >= 1, "{placement:?}: no pass ran");
        assert!(compared[0] >= 1, "{placement:?}: no chain to compare");
        // With both devices on every layer the block's first links after a pass part from
        // the reference's more often than after steps of the own rows (OPEN.md): open.
        if placement == Placement::Shared {
            continue;
        }
        assert!(
            agreed[0] + compared[0] / 5 + usize::from(compared[0] < 5 && compared[0] > 1)
                >= compared[0],
            "{placement:?}: first links agree in {} of {}",
            agreed[0],
            compared[0]
        );
    }
}

/// The MTP block follows a prompt read in wide steps whose slices start at
/// cells where its launches of eight rows do not (in the engine a hit in the
/// prompt cache leaves such a cell: the first tokens of a chat). Each launch
/// decides which of its rows write once the launches before it have given
/// the tiles they write exact slots: every cell of the prompt gets the
/// block's keys and values, the block's chains draft what the reference's
/// draft from the same cells, and no row writes into a converted entry (which
/// ended the request with "writing K and V reported status 0x4" before).
#[test]
fn the_mtp_block_follows_wide_steps_that_start_at_any_cell() {
    if !enabled() {
        return;
    }
    // Slices of 196 rows: the second starts at cell 196, its rows cross the
    // tiles at 256 and at 384 in the middle of launches.
    let prompt = prompt(420, 9);
    for placement in placements() {
        let (runtime, model) = start_drafting_unpriced_rows(1, placement, 16, 196);
        warm(&model);
        model.take_mtp_chains();
        let rx = submit(&runtime, 1, &prompt, 24);
        let answer = rx.recv_timeout(Duration::from_secs(300)).expect("an answer in time");
        let chains = model.take_mtp_chains();
        let times = model.mtp_times();
        runtime.shutdown();
        let answer = answer.unwrap_or_else(|e| panic!("{placement:?}: the request failed: {e}"));
        let (compared, agreed) = chains_against_the_reference(1, &chains, &prompt, &answer);
        eprintln!(
            "RES {placement:?}: {} chains; against the reference per link {agreed:?} of \
             {compared:?}; {} rows only wrote keys",
            chains.len(),
            times.key_rows
        );
        assert!(times.key_rows as usize >= prompt.len() - 1, "{placement:?}: {times:?}");
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

/// Drafts change no token: with the MTP block a reply is what it is
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
        let times = model.mtp_times();
        runtime.shutdown();
        eprintln!(
            "RES {placement:?}: {} tokens; draft rows {}, accepted {}, rejected {}; the block \
             made {} links in {} steps, verified per link {:?}, accepted {:?}; {} rows only \
             wrote keys",
            plain.len(),
            counters.draft_rows,
            counters.drafts_accepted,
            counters.drafts_rejected,
            counters.model_draft_links,
            counters.model_draft_steps,
            times.offered,
            times.accepted,
            times.key_rows,
        );
        assert!(plain.len() > 12, "{placement:?}: the answer is {} tokens", plain.len());
        assert!(counters.model_draft_links > 0, "{placement:?}: the block drafted nothing");
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

/// The draft vocabulary of `ARES_TEST_DRAFT_VOCAB` (a token id per line), when
/// it is set.
fn draft_vocabulary() -> Option<Vec<u32>> {
    let path = std::env::var_os("ARES_TEST_DRAFT_VOCAB")?;
    Some(ares_model::files::read_draft_vocabulary(path.as_ref()).expect("a draft vocabulary"))
}

/// The runtime over the real model on both cards (or the one there is), with
/// every miss uploaded and none computed by the CPU, whose arithmetic is not
/// the device's: what a row computes then does not depend on where its
/// experts were found. With `draft` the model drafts with that block, in
/// windows of four rows whose price is learned from the steps that run; with
/// `dflash` with that drafter as well, in windows of eight rows; `vocabulary`
/// is the head view of a draft vocabulary for the model's drafters.
fn start_flash_with(
    path: &std::path::Path,
    draft: Option<&std::path::Path>,
    dflash: Option<Arc<ares_model::dflash::File>>,
    vocabulary: Option<&[u32]>,
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
        inner = inner.with_mtp(Gguf::open(draft).unwrap()).expect("the MTP block fits");
    }
    let window = match (&dflash, draft) {
        (Some(_), _) => 8,
        (None, Some(_)) => 4,
        (None, None) => 1,
    };
    if let Some(file) = dflash {
        inner = inner.with_dflash(file).expect("the DFlash drafter fits");
    }
    if let Some(tokens) = vocabulary {
        inner = inner.with_draft_vocabulary(tokens).expect("a draft vocabulary");
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
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    // A small tier in RAM: what has no slot in VRAM is read from the file.
    gpu.ram_limit += 1 << 30;
    gpu.vram_headroom = 512 << 20;
    (Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts"), inner)
}

/// On the real model drafts change no token: Flash-Next writes with its draft
/// block, token for token, what it writes without drafts, greedy and sampled
/// with a seed, and the block's drafts are accepted. The CPU serves no miss
/// (see [`start_flash_with`]), and both replies come from one engine: two engines
/// on two cards each measure at their start how the cards share every layer
/// (the time of a read of a stream mixer on each), the share differs from
/// start to start (0.733 to 0.751 of the key heads on the faster card in eight
/// starts), and with it how the sums of a layer are cut between the cards, so
/// the comparison would be one of two arithmetics, which a near tie tells
/// apart. Needs `ARES_TEST_FLASH_NEXT` and `ARES_TEST_FLASH_NEXT_MTP`.
#[test]
fn flash_next_writes_with_drafts_what_it_writes_without() {
    if !enabled() {
        return;
    }
    let (Some(path), Some(draft)) = (
        std::env::var_os("ARES_TEST_FLASH_NEXT").map(PathBuf::from),
        std::env::var_os("ARES_TEST_FLASH_NEXT_MTP").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipped: set ARES_TEST_FLASH_NEXT to the first part of Flash-Next and \
             ARES_TEST_FLASH_NEXT_MTP to the file of its MTP block"
        );
        return;
    };
    // The whole head, and with `ARES_TEST_DRAFT_VOCAB` the head view of a
    // draft vocabulary on an engine of its own.
    let vocabulary = draft_vocabulary();
    for vocabulary in
        [None, vocabulary.as_deref()].into_iter().take(1 + vocabulary.is_some() as usize)
    {
        let view = vocabulary
            .map_or("the whole head".to_owned(), |v| format!("a draft vocabulary of {}", v.len()));
        // The replies without drafts report their probabilities, which judge a reply with
        // drafts where it parts from them.
        let greedy = Sampling {
            temperature: Some(0.0),
            logprobs: Some(LogprobRequest { top: 8, after_chain: false }),
            ..Sampling::default()
        };
        let seeded = Sampling { temperature: Some(0.8), seed: Some(5), ..Sampling::default() };
        let seeded_probs = Sampling {
            logprobs: Some(LogprobRequest { top: 8, after_chain: true }),
            ..seeded.clone()
        };
        let (runtime, model) = start_flash_with(&path, Some(&draft), None, vocabulary);
        // The replies with drafts first: a reply that repeats one the engine has
        // written is drafted by the n-gram drafter alone.
        let drafted = ask(&runtime, 1, &FLASH_PROMPT, 64);
        let (sampled, _) = ask_sampled(&runtime, 2, &FLASH_PROMPT, 48, seeded.clone(), true);
        let counters = runtime.snapshot().expect("the runtime is running").counters;
        let times = model.mtp_times();
        let (plain, plain_probs) = ask_sampled(&runtime, 3, &FLASH_PROMPT, 64, greedy, false);
        let (plain_seeded, seeded_probs) =
            ask_sampled(&runtime, 4, &FLASH_PROMPT, 48, seeded_probs, false);
        let undrafted = runtime.snapshot().expect("the runtime is running").counters.draft_rows
            - counters.draft_rows;
        let stats = model.take_store_stats().expect("the parts are open");
        runtime.shutdown();
        eprintln!(
            "RES Flash-Next, {view}: {} and {} tokens; draft rows {}, accepted {}, rejected {}; the \
         block made {} links in {} steps, {:.2} ms a link ({:.2} ms a step that drafts); \
         verified per link {:?}, accepted {:?}; prefix cache: {} hits, {} rows replayed",
            plain.len(),
            plain_seeded.len(),
            counters.draft_rows,
            counters.drafts_accepted,
            counters.drafts_rejected,
            times.links,
            times.steps,
            times.chain_ms / times.links.max(1) as f64,
            times.chain_ms / times.steps.max(1) as f64,
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
        // A window's rows are computed in other steps than the rows of a reply without
        // drafts, so a near tie may part the replies (2026-11-23: at token 16, "," against "."
        // in the RTX 4070 and RTX 3060 split, with the whole head and the draft vocabulary
        // alike). Where they part, the reply without drafts has the token of the reply with
        // drafts close to its own (greedy), or among the tokens it could have drawn (sampled).
        for (what, got, want, probs) in [
            ("a greedy reply", &drafted, &plain, &plain_probs),
            ("a sampled reply", &sampled, &plain_seeded, &seeded_probs),
        ] {
            let Some(at) = got.iter().zip(want.iter()).position(|(a, b)| a != b) else {
                assert_eq!(got.len(), want.len(), "drafts changed the length of {what}, {view}");
                continue;
            };
            let token = got[at];
            let there = probs.get(at).expect("the reply without drafts reports every token");
            let alternative = there.alternatives.iter().find(|a| a.token == token);
            let behind = alternative.map_or(f32::INFINITY, |a| there.logprob - a.logprob);
            eprintln!(
                "RES Flash-Next, {view}: with drafts {what} parts at token {at}: {token} and not \
                 {}; without drafts their log-probabilities are {:?} and {} (the alternatives \
                 {:?})",
                want[at],
                alternative.map(|a| a.logprob),
                there.logprob,
                there.alternatives.iter().map(|a| (a.token, a.logprob)).collect::<Vec<_>>(),
            );
            match what {
                "a greedy reply" => assert!(
                    behind <= NEAR_TIE_NATS,
                    "drafts changed {what} of Flash-Next, {view}: token {at} is {token}, \
                     {behind} below the token without drafts ({got:?} against {want:?})"
                ),
                _ => assert!(
                    alternative.is_some(),
                    "drafts changed {what} of Flash-Next, {view}: token {at} is {token}, which \
                     the reply without drafts could not draw ({got:?} against {want:?})"
                ),
            }
        }
    }
}

/// How far below the most likely token a token of a reply of a real model may be in the
/// log-probabilities of the same request written without drafts, and still be a near tie that
/// the rounding of other steps flips: 0.25, the two within a factor of 1.3 of each other.
const NEAR_TIE_NATS: f32 = 0.25;

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
        std::env::var_os("ARES_TEST_FLASH_NEXT").map(PathBuf::from),
        std::env::var_os("ARES_TEST_DFLASH_FLASHNEXT").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipped: set ARES_TEST_FLASH_NEXT to the first part of Flash-Next and \
             ARES_TEST_DFLASH_FLASHNEXT to its DFlash2 file"
        );
        return;
    };
    let file = Arc::new(ares_model::dflash::File::open(&file).unwrap());
    let vocabulary = draft_vocabulary();
    for vocabulary in
        [None, vocabulary.as_deref()].into_iter().take(1 + vocabulary.is_some() as usize)
    {
        let view = vocabulary
            .map_or("the whole head".to_owned(), |v| format!("a draft vocabulary of {}", v.len()));
        let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
        let (runtime, model) = start_flash_with(&path, None, Some(file.clone()), vocabulary);
        // The reply with drafts first, as above.
        let drafted = ask(&runtime, 1, &FLASH_PROMPT, 64);
        let counters = runtime.snapshot().expect("the runtime is running").counters;
        let (plain, _) = ask_sampled(&runtime, 2, &FLASH_PROMPT, 64, greedy, false);
        let undrafted = runtime.snapshot().expect("the runtime is running").counters.draft_rows
            - counters.draft_rows;
        let stats = model.take_store_stats().expect("the parts are open");
        runtime.shutdown();
        eprintln!(
            "RES Flash-Next with DFlash, {view}: {} tokens in {} steps, {:.2} ms a step that only \
         drafts ({} of them); tested by position {:?}, accepted {:?}; prefix cache: {} hits, {} \
         cells cached, {} rows replayed, {} checkpoints restored",
            counters.tokens,
            counters.steps,
            counters.model_draft_us as f64 / 1e3 / counters.model_draft_steps.max(1) as f64,
            counters.model_draft_steps,
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
        assert_eq!(drafted, plain, "DFlash drafts changed a greedy reply of Flash-Next, {view}");
    }
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
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
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

/// The MTP block in the model file (what `--drafters mtp` reads) drafts as
/// the block of a file of its own does, and with a draft vocabulary every
/// draft is a token of it (the head view, which the model's own head reads in
/// its order too); drafts change no token either way.
#[test]
fn the_mtp_block_of_the_model_file_and_a_draft_vocabulary_change_no_token() {
    if !enabled() {
        return;
    }
    let seed = 2;
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4rt-{seed}-mtp.gguf"));
    std::fs::write(&path, model(seed).bytes_with_mtp((GgmlType::IQ2_S, GgmlType::Q2_0)))
        .expect("the test directory is writable");
    let mut repeats = prompt(12, 5);
    repeats.extend_from_within(..);
    let placement = Placement::One;
    let (runtime, plain_model) = start(seed, placement, 16, RuntimeConfig::default());
    warm(&plain_model);
    let plain = ask(&runtime, 1, &repeats, 80);
    runtime.shutdown();
    drop(plain_model);

    // The block of the model file.
    let config = RuntimeConfig { window_rows: 8, ..RuntimeConfig::default() };
    let (runtime, drafting) =
        start_with(&path, Some(&path), placement, 16, config.clone(), MAX_ROWS, Cache::On);
    warm(&drafting);
    let with_block = ask(&runtime, 1, &repeats, 80);
    let counters = runtime.snapshot().expect("the runtime is running").counters;
    runtime.shutdown();
    drop(drafting);
    eprintln!(
        "RES the block of the model file: {} links in {} steps, draft rows {}, accepted {}",
        counters.model_draft_links,
        counters.model_draft_steps,
        counters.draft_rows,
        counters.drafts_accepted
    );
    assert!(counters.model_draft_links > 0, "the block of the model file drafted nothing");
    assert_eq!(with_block, plain, "the block of the model file changed the text");

    // A draft vocabulary of every third token.
    let vocabulary: Vec<u32> = (0..model(seed).shape.n_vocab as u32).step_by(3).collect();
    let devices = open_devices(placement);
    let gguf = Gguf::open(gguf_path(seed)).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let opts = options(placement, n_layer, 16, MAX_ROWS, MAX_ROWS);
    let inner = Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS)
        .unwrap()
        .with_mtp(Gguf::open(draft_path(seed)).unwrap())
        .expect("the MTP block fits")
        .with_draft_vocabulary(&vocabulary)
        .expect("a draft vocabulary");
    let inner = Arc::new(inner);
    let config = RuntimeConfig {
        pages: 16,
        cells_per_page: 128,
        max_slots: SLOTS,
        window_rows: 8,
        ..config
    };
    let model: Arc<dyn Model> = Arc::new(made(&inner, Cache::On));
    let mut gpu = GpuConfig::new(&config, &*model);
    gpu.ram_limit += 256 << 20;
    gpu.vram_headroom = 512 << 20;
    let runtime = Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts");
    warm(&inner);
    inner.take_mtp_chains();
    let with_vocabulary = ask(&runtime, 1, &repeats, 80);
    let chains = inner.take_mtp_chains();
    let report = inner.mtp_report().expect("the model has an MTP block");
    runtime.shutdown();
    let drafts: Vec<u32> = chains.iter().flat_map(|c| c.drafts.iter().copied()).collect();
    eprintln!(
        "RES a draft vocabulary of {} tokens: {} chains, {} drafts; the block keeps {} rows of \
         the embedding table ({} B)",
        vocabulary.len(),
        chains.len(),
        drafts.len(),
        report.kept_rows,
        report.kept_bytes
    );
    assert_eq!(report.kept_rows as usize, vocabulary.len());
    assert!(!drafts.is_empty(), "the block drafted nothing with a draft vocabulary");
    assert!(drafts.iter().all(|t| vocabulary.contains(t)), "a draft outside the vocabulary");
    assert_eq!(with_vocabulary, plain, "drafts from a draft vocabulary changed the text");
}
