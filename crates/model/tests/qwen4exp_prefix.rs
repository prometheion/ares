//! The prefix cache of `qwen4exp` on real devices: a prompt that continues
//! text the engine has computed starts from its pages and from the
//! checkpoints of the state per sequence (the recurrent state, the carried
//! keys of the indexers, the n-gram history and what the n-gram feed knows of
//! the sequence, the MTP block's carry and keys) instead of reading
//! everything again, from the cards, from RAM and from SSD.
//!
//! A made-up model of the architecture runs through the runtime on the GPUs,
//! in every placement: the layers on one device, a range of layers on each of
//! two, and both devices on every layer. Every expert has a slot in VRAM, so
//! no step leaves one to the CPU, and a request is answered the same way
//! whichever steps compute it: every test runs the same requests on two
//! engines that differ in one thing, `Model::prefix_cache` (and with it the
//! levels below the cards), and the answers must be the same tokens.
//!
//! The tests open GPUs and are skipped unless `ARES_TEST_CUDA=1` is set; the
//! placements over two devices need two. The model's vocabulary is a
//! placeholder, so the prompts are the text `t5 t9 ...` and the answers are
//! read back the same way. Lines that start with `RES` are measurements.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Sampling, Usage};
use ares_core::GgmlType;
use ares_cuda::{CacheFormat, Device, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen4exp::Config;
use ares_model::qwen4exp::device::{ModelOptions, Options, Qwen4Exp, Spread};
use ares_model::qwen4exp::synthetic::{NgramShape, Shape, Synthetic, Types};
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig, Snapshot, SsdConfig};
use common::gpu::enabled;
use common::model::{Wrapped, byte_tokenizer};

mod common;

const MAX_ROWS: u32 = 8;
const PAGE: u32 = 128;
const CELLS: u32 = 1024;
const SLOTS: u32 = 2;
/// Pages of the pool unless a test says otherwise: room for the sequences of
/// a test and for what the cache keeps of them on the cards.
const PAGES: u32 = 24;

/// A made-up model with every part of the architecture at sizes the kernels
/// take (the one of `qwen4exp_runtime.rs`).
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
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4px-{seed}.gguf"));
    if !path.exists() {
        let bytes = model(seed).bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0);
        std::fs::write(&path, bytes).expect("the test directory is writable");
    }
    path
}

/// The file of a made-up MTP block for the made-up model of `seed`.
fn draft_path(seed: u64) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4px-{seed}-draft.gguf"));
    if !path.exists() {
        let bytes = model(seed).draft_bytes(Some((GgmlType::IQ2_S, GgmlType::Q2_0)));
        std::fs::write(&path, bytes).expect("the test directory is writable");
    }
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
    } else {
        eprintln!("one GPU: the placements over two are not tested");
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
    kv: (CacheFormat, CacheFormat),
    pages: u32,
) -> ModelOptions {
    ModelOptions {
        part: Options {
            kv,
            max_rows: MAX_ROWS,
            logit_rows: MAX_ROWS,
            window_rows: MAX_ROWS,
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

/// How an engine of a test is made.
#[derive(Clone)]
struct Setup {
    seed: u64,
    placement: Placement,
    /// With the model's prefix cache.
    cached: bool,
    /// With the model's MTP block, drafting in windows of up to eight rows.
    draft: bool,
    kv: (CacheFormat, CacheFormat),
    /// Checkpoints the prompt cache has room for in RAM.
    checkpoints: u64,
    /// The prompt cache's level on SSD.
    ssd: Option<SsdConfig>,
    /// Pages of the pool.
    pages: u32,
    /// A RAM limit that leaves the tier of experts less than it wants, so
    /// that it takes everything the prompt cache does not keep whatever the
    /// tier holds.
    tier_takes_all: bool,
}

impl Setup {
    fn new(seed: u64, placement: Placement, cached: bool) -> Self {
        Self {
            seed,
            placement,
            cached,
            draft: false,
            kv: (CacheFormat::Q8, CacheFormat::Q8),
            checkpoints: 16,
            ssd: None,
            pages: PAGES,
            tier_takes_all: false,
        }
    }
}

/// An engine on the made-up model.
struct Rig {
    runtime: Runtime,
    model: Arc<Qwen4Exp>,
    next: AtomicU64,
}

fn rig(setup: &Setup) -> Rig {
    let devices = open_devices(setup.placement);
    let gguf = Gguf::open(gguf_path(setup.seed)).unwrap();
    let n_layer = Config::from_gguf(&gguf).unwrap().n_layer;
    let opts = options(setup.placement, n_layer, setup.kv, setup.pages);
    let mut inner =
        Qwen4Exp::new(gguf, byte_tokenizer(None), "made", &devices, opts, CELLS).unwrap();
    if setup.draft {
        inner = inner
            .with_mtp(Gguf::open(draft_path(setup.seed)).unwrap())
            .expect("the MTP block fits");
    }
    let inner = Arc::new(inner);
    let spec = inner.prefix_cache().expect("qwen4exp takes part");
    let checkpoint = spec.bytes();
    let config = RuntimeConfig {
        pages: setup.pages,
        cells_per_page: PAGE,
        max_slots: SLOTS,
        window_rows: if setup.draft { MAX_ROWS } else { 1 },
        // Checkpoints inside a prompt and while generating are as dense as a
        // page, so that a short test crosses several of them.
        checkpoint_prompt_interval: PAGE,
        checkpoint_generation_interval: PAGE,
        cache_ram: setup.checkpoints * checkpoint + if setup.ssd.is_some() { 64 << 20 } else { 0 },
        ssd: setup.ssd.clone(),
        ..RuntimeConfig::default()
    };
    let model: Arc<dyn Model> = Arc::new(Wrapped::new(inner.clone()).words().cache(setup.cached));
    let mut gpu = GpuConfig::new(&config, &*model);
    if setup.tier_takes_all {
        // Less than the tier wants (at least 64 MiB) next to what the
        // executor needs, and nothing named for the prompt cache.
        gpu.ram_limit = gpu.ram_limit - config.cache_ram + (32 << 20);
    } else {
        // Room for the tier of experts next to what the executor needs.
        gpu.ram_limit += 256 << 20;
    }
    gpu.vram_headroom = 512 << 20;
    let runtime = Runtime::start_gpu(model, devices, config, gpu).expect("the runtime starts");
    warm(&inner);
    Rig { runtime, model: inner, next: AtomicU64::new(1) }
}

/// Brings every expert into a slot, so that no step leaves one to the CPU
/// (whose arithmetic is not the device's: a made-up model has near ties).
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

/// What a request said and what it took from the cache.
#[derive(Clone, Debug)]
struct Answer {
    tokens: Vec<u32>,
    usage: Usage,
}

impl Rig {
    fn submit(&self, prompt: &[u32], max_tokens: u32) -> mpsc::Receiver<Result<Answer, String>> {
        let (tx, rx) = mpsc::channel();
        let text = Arc::new(Mutex::new(String::new()));
        let sink = EventSink::new(move |event| {
            match event {
                Event::Text(piece) => text.lock().unwrap().push_str(&piece),
                Event::Done { usage, .. } => {
                    let tokens = text
                        .lock()
                        .unwrap()
                        .split_whitespace()
                        .map(|w| w[1..].parse().unwrap())
                        .collect();
                    let _ = tx.send(Ok(Answer { tokens, usage }));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
                Event::Logprobs(_) | Event::Progress(_) | Event::Generated(_) => {}
            }
            true
        });
        let words: Vec<String> = prompt.iter().map(|t| format!("t{t}")).collect();
        self.runtime
            .submit(
                GenerateRequest {
                    model: "made".into(),
                    prompt: Prompt::Text(words.join(" ")),
                    sampling: Sampling { temperature: Some(0.0), ..Sampling::default() },
                    max_tokens: Some(max_tokens),
                    stop: Vec::new(),
                    draft: true,
                    cache: true,
                    structure: None,
                    priority: 0,
                    id: RequestId(self.next.fetch_add(1, Ordering::Relaxed)),
                },
                sink,
            )
            .unwrap();
        rx
    }

    fn ask(&self, prompt: &[u32], max_tokens: u32) -> Answer {
        self.submit(prompt, max_tokens)
            .recv_timeout(Duration::from_secs(300))
            .expect("the engine answers")
            .expect("the request succeeds")
    }

    fn snapshot(&self) -> Snapshot {
        self.runtime.snapshot().expect("the engine runs")
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.runtime.shutdown();
    }
}

/// A prompt of `n` tokens of the vocabulary, from a seed.
fn prompt(n: usize, seed: u64) -> Vec<u32> {
    Shape::small().tokens(n, seed).into_iter().map(|t| t % 80 + 10).collect()
}

#[track_caller]
fn assert_same(what: &str, cold: &Answer, cached: &Answer) {
    assert_eq!(cold.tokens, cached.tokens, "{what}: the answer with the cache is another");
}

fn report(what: &str, placement: Placement, answer: &Answer) {
    eprintln!(
        "RES {placement:?} {what}: {} prompt tokens, {} from the cache, {} generated",
        answer.usage.prompt_tokens,
        answer.usage.cached_prompt_tokens,
        answer.usage.completion_tokens
    );
}

fn report_counters(placement: Placement, rig: &Rig) {
    let c = rig.snapshot().counters;
    eprintln!(
        "RES {placement:?}: {} prefix hits, {} checkpoints saved, {} restored, {} rows replayed, \
         {} rows computed, {} page copies; RAM: {} pages in, {} read back; SSD: {} pages and {} \
         checkpoints written, {} pages and {} checkpoints read",
        c.prefix_hits,
        c.checkpoints_saved,
        c.checkpoints_restored,
        c.replay_rows,
        c.prefill_rows,
        c.page_copies,
        c.ram.pages_in,
        c.ram.pages_read,
        c.ssd.pages_written,
        c.ssd.states_written,
        c.ssd.pages_read,
        c.ssd.states_read,
    );
}

#[test]
fn the_same_prompt_twice_is_answered_alike_and_the_second_time_from_the_cache() {
    if !enabled() {
        return;
    }
    // Two pages and a bit, so that checkpoints lie inside the prompt.
    let text = prompt(300, 1);
    for placement in placements() {
        let cold = rig(&Setup::new(1, placement, false)).ask(&text, 24);
        let cached = rig(&Setup::new(1, placement, true));
        let first = cached.ask(&text, 24);
        let second = cached.ask(&text, 24);
        report("first", placement, &first);
        report("second", placement, &second);
        report_counters(placement, &cached);
        assert_same("the first answer", &cold, &first);
        assert_same("the second answer", &cold, &second);
        assert_eq!(first.usage.cached_prompt_tokens, 0);
        assert!(
            second.usage.cached_prompt_tokens >= 2 * PAGE,
            "{placement:?}: the full pages of the prompt come from the cache: {:?}",
            second.usage
        );
        let c = cached.snapshot().counters;
        assert!(c.prefix_hits >= 1 && c.checkpoints_saved >= 1 && c.checkpoints_restored >= 1);
        assert!(c.page_copies >= 1, "the tail of the first run is copied before the second writes");
        assert!(
            c.replay_rows > 0,
            "the second prompt replays from its checkpoint to the tail's end"
        );
        assert!(cached.snapshot().checkpoint_ram > 0, "a checkpoint was kept in locked memory");
        assert_eq!(cached.model.take_store_stats().map(|s| s.picks == s.hits), Some(true));
    }
}

/// The turns of a conversation: each prompt is the one before, its answer and
/// a new part.
fn conversation(rig: &Rig, turns: usize, answer: u32) -> Vec<Answer> {
    let mut text = prompt(200, 2);
    let mut answers = Vec::new();
    for turn in 0..turns {
        let said = rig.ask(&text, answer);
        text.extend_from_slice(&said.tokens);
        text.extend(prompt(40 + 30 * turn, 10 + turn as u64));
        answers.push(said);
    }
    answers
}

#[test]
fn a_conversation_continued_over_several_turns_is_answered_as_without_a_cache() {
    if !enabled() {
        return;
    }
    for placement in placements() {
        let cold = conversation(&rig(&Setup::new(2, placement, false)), 3, 20);
        let cached = rig(&Setup::new(2, placement, true));
        let warm = conversation(&cached, 3, 20);
        for (turn, (cold, warm)) in cold.iter().zip(&warm).enumerate() {
            report(&format!("turn {turn}"), placement, warm);
            assert_same(&format!("turn {turn}"), cold, warm);
        }
        report_counters(placement, &cached);
        for (turn, said) in warm.iter().enumerate().skip(1) {
            // The turn before ended in a tail whose state is its checkpoint:
            // everything of it but its last token comes from the cache.
            let before = warm[turn - 1].usage;
            assert!(
                said.usage.cached_prompt_tokens + 1
                    >= before.prompt_tokens + before.completion_tokens,
                "{placement:?} turn {turn}: {:?} after {before:?}",
                said.usage
            );
        }
    }
}

/// A RAM limit that the tier of experts takes whole, with no level on SSD:
/// the prompt cache still keeps one checkpoint place, booked before the tier,
/// so each turn of a conversation resumes from a checkpoint of the turn before
/// and replays only the rows after it, where without the place every hit
/// replayed every row it found.
#[test]
fn a_conversation_resumes_from_a_checkpoint_when_the_tier_takes_all_the_ram() {
    if !enabled() {
        return;
    }
    for placement in placements() {
        let cold = conversation(&rig(&Setup::new(2, placement, false)), 3, 20);
        let cached = rig(&Setup { tier_takes_all: true, ..Setup::new(2, placement, true) });
        let warm = conversation(&cached, 3, 20);
        for (turn, (cold, warm)) in cold.iter().zip(&warm).enumerate() {
            report(&format!("tier takes all, turn {turn}"), placement, warm);
            assert_same(&format!("turn {turn}"), cold, warm);
        }
        report_counters(placement, &cached);
        let c = cached.snapshot().counters;
        let found: u64 = warm[1..].iter().map(|a| u64::from(a.usage.cached_prompt_tokens)).sum();
        assert!(found > 0, "{placement:?}: the later turns found the earlier ones");
        assert!(c.checkpoints_restored >= 2, "{placement:?}: {} restored", c.checkpoints_restored);
        // Not a full replay: without the place every row found was replayed.
        assert!(
            c.replay_rows < found,
            "{placement:?}: {} rows replayed of {found} found",
            c.replay_rows
        );
    }
}

#[test]
fn a_prompt_that_parts_from_a_cached_one_inside_a_page_is_answered_as_without_a_cache() {
    if !enabled() {
        return;
    }
    let first = prompt(300, 3);
    // The same first 200 tokens: a page and 72 cells of the next.
    let mut second = first[..200].to_vec();
    second.extend(prompt(120, 4));
    for placement in placements() {
        let cold = rig(&Setup::new(3, placement, false)).ask(&second, 24);
        let cached = rig(&Setup::new(3, placement, true));
        cached.ask(&first, 8);
        let warm = cached.ask(&second, 24);
        report("parted", placement, &warm);
        report_counters(placement, &cached);
        assert_same("the prompt that parts inside a page", &cold, &warm);
        assert!(warm.usage.cached_prompt_tokens >= PAGE, "{:?}", warm.usage);
    }
}

#[test]
fn a_sequence_that_drafts_with_the_mtp_block_resumes_with_its_carry_and_its_keys() {
    if !enabled() {
        return;
    }
    for placement in placements() {
        // Without drafts and without the cache, against drafts and the cache:
        // drafts change no token of this model while every expert has a slot.
        let cold = conversation(&rig(&Setup::new(5, placement, false)), 3, 32);
        let cached = rig(&Setup { draft: true, ..Setup::new(5, placement, true) });
        let warm = conversation(&cached, 3, 32);
        for (turn, (cold, warm)) in cold.iter().zip(&warm).enumerate() {
            report(&format!("turn {turn} with drafts"), placement, warm);
            assert_same(&format!("turn {turn} with drafts"), cold, warm);
        }
        report_counters(placement, &cached);
        let c = cached.snapshot().counters;
        eprintln!("RES {placement:?}: {} drafts, {} accepted", c.draft_rows, c.drafts_accepted);
        assert!(c.checkpoints_restored >= 2 && c.draft_rows > 0, "{placement:?}");
    }
}

#[test]
fn the_formats_with_a_tile_mean_keep_their_exact_pages_through_the_cache() {
    if !enabled() {
        return;
    }
    let q4 = (CacheFormat::Q4, CacheFormat::Q4);
    let text = prompt(400, 6);
    for placement in placements() {
        let cold = rig(&Setup { kv: q4, ..Setup::new(6, placement, false) }).ask(&text, 24);
        let cached = rig(&Setup { kv: q4, ..Setup::new(6, placement, true) });
        cached.ask(&text, 24);
        let warm = cached.ask(&text, 24);
        report("q4", placement, &warm);
        report_counters(placement, &cached);
        assert_same("K and V in q4", &cold, &warm);
    }
}

#[test]
fn pages_that_left_the_cards_come_back_from_ram() {
    if !enabled() {
        return;
    }
    let a = prompt(600, 7);
    let b = prompt(1000, 8);
    // A pool that holds one of the conversations and not both.
    let small = |placement, cached| Setup { pages: 12, ..Setup::new(7, placement, cached) };
    for placement in placements() {
        let cold = rig(&small(placement, false));
        let first = cold.ask(&a, 16);
        let mut a2 = a.clone();
        a2.extend_from_slice(&first.tokens);
        a2.extend(prompt(30, 9));
        let expected = cold.ask(&a2, 16);
        drop(cold);
        let cached = rig(&small(placement, true));
        assert_same("the first turn", &first, &cached.ask(&a, 16));
        // The other conversation pushes the first out of the cards.
        cached.ask(&b, 8);
        let before = cached.snapshot().counters.ram;
        let second = cached.ask(&a2, 16);
        let after = cached.snapshot().counters.ram;
        report("from RAM", placement, &second);
        report_counters(placement, &cached);
        assert_same("the second turn, from RAM", &expected, &second);
        assert!(after.pages_read > before.pages_read, "{placement:?}: {before:?} {after:?}");
        assert!(second.usage.cached_prompt_tokens >= 4 * PAGE, "{:?}", second.usage);
    }
}

/// A fresh directory for a store, under the build's directory for tests.
fn store_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4px-store-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_conversation_resumes_from_ssd_after_the_engine_ends() {
    if !enabled() {
        return;
    }
    let a = prompt(600, 11);
    for placement in placements() {
        let dir = store_dir(&format!("{placement:?}"));
        let ssd = SsdConfig { min_cells: PAGE, ..SsdConfig::new(&dir, 1 << 30) };
        let cold = rig(&Setup::new(11, placement, false));
        let first = cold.ask(&a, 16);
        let mut a2 = a.clone();
        a2.extend_from_slice(&first.tokens);
        a2.extend(prompt(30, 12));
        let expected = cold.ask(&a2, 16);
        drop(cold);
        {
            let writer = rig(&Setup { ssd: Some(ssd.clone()), ..Setup::new(11, placement, true) });
            assert_same("the first turn", &first, &writer.ask(&a, 16));
            report_counters(placement, &writer);
        }
        let reader = rig(&Setup { ssd: Some(ssd.clone()), ..Setup::new(11, placement, true) });
        let held = reader.snapshot().counters.ssd;
        assert!(held.store_pages >= 4 && held.store_states >= 1, "{placement:?}: {held:?}");
        let second = reader.ask(&a2, 16);
        report("from SSD", placement, &second);
        report_counters(placement, &reader);
        assert_same("the second turn, from SSD", &expected, &second);
        let read = reader.snapshot().counters.ssd;
        assert!(read.pages_read >= 4 && read.states_read >= 1, "{placement:?}: {read:?}");
        assert!(second.usage.cached_prompt_tokens >= 4 * PAGE, "{:?}", second.usage);
        drop(reader);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn without_checkpoints_a_hit_replays_every_cached_row_and_answers_alike() {
    if !enabled() {
        return;
    }
    let text = prompt(300, 13);
    for placement in placements() {
        let cold = rig(&Setup::new(13, placement, false)).ask(&text, 16);
        let cached = rig(&Setup { checkpoints: 0, ..Setup::new(13, placement, true) });
        cached.ask(&text, 16);
        let second = cached.ask(&text, 16);
        report("no checkpoints", placement, &second);
        report_counters(placement, &cached);
        assert_same("a hit without checkpoints", &cold, &second);
        let c = cached.snapshot().counters;
        assert_eq!(c.checkpoints_saved, 0);
        assert!(c.replay_rows >= 2 * u64::from(PAGE), "{placement:?}: {} replayed", c.replay_rows);
    }
}
