//! The prefix cache on real devices, with the real Qwen3.5-0.8B: a prompt that
//! continues text the engine has computed starts from its pages and from the
//! checkpoints of its recurrent state instead of reading everything again.
//!
//! Every test runs the same requests on two engines that differ in one thing,
//! `Model::prefix_cache`: one has the cache and one does not. The model
//! answers at once, without thinking aloud, and the requests ask it to copy a
//! text, whose words follow with a wide margin; with the cache the answer must
//! be the same text as without, token for token. It is run on one card (each
//! of them), on a tensor split and on a layer split; the splits need two
//! cards. The prompt steps of the splits are cut into groups of 32 rows, so
//! that the rows of a replay are cut into groups too.
//!
//! The products are exact (`Precision::Exact`), and still a row's result
//! depends on the size of the step that computes it, by rounding (the last
//! test but one measures how much), and the cache changes the steps that
//! compute a prompt. So the logits of a run from the cache are not those of a
//! run without it bit for bit (`the_picks_of_a_request_...` reports and bounds
//! the difference), and a text with a near tie could part at it. Two requests
//! that decode together instead of alone are such a change too, so the tests
//! with two requests at once run without drafts and compare each request with
//! its answer alone. What the device part does is checked bit for bit, with
//! steps of the same sizes, in `qwen35_prefix_part.rs` and
//! `qwen35_prefix_mtp.rs`.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL`. Lines that start
//! with `RES` are measurements.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{
    Engine as _, EngineError, Event, EventSink, FinishReason, GenerateRequest, Message, ModelCard,
    Prompt, RequestId, Role, Sampling, Usage,
};
use ares_cuda::{Device, Precision, SampleParams, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Rows, Session, Spread, Wanted};
use ares_runtime::{
    Clock, ContextSpec, DeviceContext, DeviceModel, ExecError, Executor, GpuConfig, GpuExecutor,
    LinkSpec, Model, ModelError, Pick, PrefixCache, Runtime, RuntimeConfig, Snapshot, Split,
    StepMeta, StepOutput, WakeCounts, Woke,
};
use ares_state::PageLayout;
use ares_tokenizer::ChatOptions;

const PAGE: u32 = 128;

fn model_path() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    match std::env::var_os("ARES_TEST_QWEN35_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to run this test");
            None
        }
    }
}

/// Where the model goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// Everything on one card.
    Card(u32),
    Tensor,
    Layer,
}

impl Layout {
    fn spread(self) -> Spread {
        match self {
            Layout::Card(_) => Spread::Single,
            Layout::Tensor => Spread::Tensor(vec![0.625, 0.375]),
            Layout::Layer => Spread::Layer(vec![0.5, 0.5]),
        }
    }

    fn ordinals(self) -> Vec<u32> {
        match self {
            Layout::Card(n) => vec![n],
            _ => vec![0, 1],
        }
    }
}

/// The layouts this machine can run.
fn layouts() -> Vec<Layout> {
    let count = device_count().unwrap();
    let mut all = vec![Layout::Card(0)];
    if count >= 2 {
        all.extend([Layout::Card(1), Layout::Tensor, Layout::Layer]);
    } else {
        eprintln!("one GPU: the second card and the splits are not tested");
    }
    all
}

/// A model that is another one without its prefix cache: the engine on top of
/// it computes every prompt from its first token.
struct Uncached(Arc<Qwen35>);

impl Model for Uncached {
    fn card(&self) -> ModelCard {
        self.0.card()
    }
    fn vocab(&self) -> u32 {
        self.0.vocab()
    }
    fn ends(&self, token: u32) -> bool {
        self.0.ends(token)
    }
    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        self.0.encode(prompt)
    }
    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        self.0.piece(token, text);
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
    fn max_rows(&self) -> u32 {
        self.0.max_rows()
    }
    fn logit_rows(&self) -> u32 {
        self.0.logit_rows()
    }
    fn draft_rows(&self) -> u32 {
        self.0.draft_rows()
    }
    fn sampling(&self) -> SampleParams {
        self.0.sampling()
    }
    fn attention_span(&self) -> u32 {
        self.0.attention_span()
    }
    fn own_drafts(&self) -> u32 {
        self.0.own_drafts()
    }
    fn prefix_cache(&self) -> Option<PrefixCache> {
        None
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

/// No drafts: a request's steps are then the same whatever else runs, so that
/// two requests at once are answered as they are alone.
fn no_drafts(config: &mut RuntimeConfig, _: u64) {
    config.speculation = false;
}

/// What the test changes in the engine's settings: the bytes of a checkpoint
/// are given, so that a budget can be a number of them.
type Tweak<'a> = &'a dyn Fn(&mut RuntimeConfig, u64);

/// The executor of the devices, remembering the picks of every step: the token
/// and the logit of each row whose logits were wanted.
struct Recording {
    inner: GpuExecutor,
    picks: Arc<Mutex<Vec<Pick>>>,
}

impl Executor for Recording {
    fn devices(&self) -> usize {
        self.inner.devices()
    }
    fn clock(&self) -> Clock {
        self.inner.clock()
    }
    fn record(&mut self) -> &mut [u8] {
        self.inner.record()
    }
    fn dispatch(&mut self, meta: &StepMeta) -> Result<(), ExecError> {
        self.inner.dispatch(meta)
    }
    fn collect(&mut self, spin: Duration, out: &mut StepOutput) -> Result<Woke, ExecError> {
        let woke = self.inner.collect(spin, out)?;
        self.picks.lock().unwrap().extend_from_slice(&out.picks);
        Ok(woke)
    }
    fn device_wakes(&self) -> WakeCounts {
        self.inner.device_wakes()
    }
    fn checkpoint_ram(&self) -> u64 {
        self.inner.checkpoint_ram()
    }
    fn shutdown(&mut self) {
        self.inner.shutdown();
    }
}

/// An engine on the 0.8B.
struct Rig {
    runtime: Runtime,
    next: AtomicU64,
    /// The bytes of one checkpoint on all devices.
    checkpoint: u64,
    /// The picks of every step since the engine was ready.
    picks: Arc<Mutex<Vec<Pick>>>,
}

fn rig(path: &PathBuf, layout: Layout, cached: bool, tweak: Tweak<'_>) -> Rig {
    let options = Options {
        precision: Precision::Exact,
        max_rows: 128,
        logit_rows: 16,
        window_rows: 4,
        pipeline_rows: 32,
        ..Options::default()
    };
    // The model answers at once, without thinking aloud: a text to copy has
    // words that follow with a wide margin, which the rounding of a row does
    // not tip.
    let chat = ChatOptions { enable_thinking: false, ..ChatOptions::default() };
    let model = Arc::new(
        Qwen35::new(Gguf::open(path).unwrap(), "q", &layout.spread(), options, 4096)
            .unwrap()
            .with_chat_options(chat),
    );
    let spec = model.prefix_cache().expect("qwen35 takes part");
    let checkpoint = spec.bytes();
    let reports_of = model.clone();
    let mut config = RuntimeConfig {
        pages: 64,
        cells_per_page: PAGE,
        max_slots: 2,
        speculation: true,
        max_draft: 3,
        // Checkpoints inside a prompt and while generating are as dense as a
        // page, so that a short test crosses several of them.
        checkpoint_prompt_interval: PAGE,
        checkpoint_generation_interval: PAGE,
        checkpoint_budget: 16 * checkpoint,
        ..RuntimeConfig::default()
    };
    tweak(&mut config, checkpoint);
    let model: Arc<dyn Model> = if cached { model } else { Arc::new(Uncached(model)) };
    let gpu = GpuConfig::new(&config, &*model);
    let devices = layout
        .ordinals()
        .into_iter()
        .map(|ordinal| Device::open(ordinal, Wait::Block).unwrap())
        .collect();
    let picks = Arc::new(Mutex::new(Vec::new()));
    let executor = Recording {
        inner: GpuExecutor::start(model.clone(), devices, gpu).unwrap(),
        picks: picks.clone(),
    };
    let runtime = Runtime::start(model, Box::new(executor), config).unwrap();
    // What the model says a checkpoint takes on each device is what its parts
    // took when they were opened.
    let parts: Vec<u64> = reports_of.reports().iter().map(|r| r.checkpoint_bytes).collect();
    assert_eq!(parts, spec.checkpoint_bytes, "{layout:?}");
    // The steps of the warm-up are not the tests'.
    picks.lock().unwrap().clear();
    Rig { runtime, next: AtomicU64::new(1), checkpoint, picks }
}

/// What a request came to.
#[derive(Clone, Debug)]
struct Answer {
    text: String,
    usage: Usage,
    finish: FinishReason,
}

impl Rig {
    fn submit(&self, messages: &[Message], max_tokens: u32) -> mpsc::Receiver<Answer> {
        let (tx, rx) = mpsc::channel();
        let text = Arc::new(Mutex::new(String::new()));
        let sink = EventSink::new(move |event| {
            match event {
                Event::Text(piece) => text.lock().unwrap().push_str(&piece),
                Event::Done { finish, usage } => {
                    let text = std::mem::take(&mut *text.lock().unwrap());
                    let _ = tx.send(Answer { text, usage, finish });
                }
                Event::Error(error) => panic!("the engine failed a request: {error}"),
            }
            true
        });
        self.runtime
            .submit(
                GenerateRequest {
                    model: "q".into(),
                    prompt: Prompt::Chat { messages: messages.to_vec(), tools: None },
                    sampling: Sampling { temperature: Some(0.0), ..Sampling::default() },
                    max_tokens: Some(max_tokens),
                    stop: Vec::new(),
                    id: RequestId(self.next.fetch_add(1, Ordering::Relaxed)),
                },
                sink,
            )
            .unwrap();
        rx
    }

    fn ask(&self, messages: &[Message], max_tokens: u32) -> Answer {
        self.submit(messages, max_tokens)
            .recv_timeout(Duration::from_secs(180))
            .expect("the engine answers")
    }

    fn snapshot(&self) -> Snapshot {
        self.runtime.snapshot().expect("the engine runs")
    }
}

/// Text of about `sentences * 12` tokens that no other seed repeats.
fn passage(seed: u64, sentences: usize) -> String {
    const SUBJECTS: [&str; 8] = [
        "The old harbour master",
        "A tired courier",
        "Our youngest engineer",
        "The village baker",
        "Her grandmother",
        "That quiet clockmaker",
        "A visiting botanist",
        "The night watchman",
    ];
    const VERBS: [&str; 8] =
        ["repaired", "counted", "remembered", "painted", "carried", "measured", "forgot", "sold"];
    const OBJECTS: [&str; 8] = [
        "seventeen brass lanterns",
        "a map of the northern marshes",
        "three crates of green apples",
        "the broken weather vane",
        "every letter from the south",
        "two hundred paper boats",
        "a cabinet full of old keys",
        "the last train timetable",
    ];
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    (0..sentences)
        .map(|i| {
            format!(
                "{} {} {} on day {}.",
                SUBJECTS[next(8)],
                VERBS[next(8)],
                OBJECTS[next(8)],
                i * 3 + next(7)
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn user(text: impl Into<String>) -> Message {
    Message::new(Role::User, text)
}

/// A request to say a text again.
fn copy(text: String) -> String {
    format!("{text}\n\nCopy the text above word for word.")
}

/// Fails with both texts and the place where they part.
#[track_caller]
fn assert_same(what: &str, cold: &Answer, cached: &Answer) {
    let same = cold.text.chars().zip(cached.text.chars()).take_while(|(a, b)| a == b).count();
    assert_eq!(
        cold.text, cached.text,
        "{what}: the answer with the cache parts from the one without after {same} characters\n\
         --- without ---\n{}\n--- with ---\n{}",
        cold.text, cached.text
    );
    assert_eq!(cold.usage.completion_tokens, cached.usage.completion_tokens, "{what}");
    assert_eq!(cold.finish, cached.finish, "{what}");
}

fn report(what: &str, layout: Layout, answer: &Answer) {
    eprintln!(
        "RES {layout:?} {what}: {} prompt tokens, {} from the cache, {} generated",
        answer.usage.prompt_tokens,
        answer.usage.cached_prompt_tokens,
        answer.usage.completion_tokens
    );
}

#[test]
fn the_same_prompt_twice_is_answered_alike_and_the_second_time_from_the_cache() {
    let Some(path) = model_path() else { return };
    // Six pages and a bit, so that checkpoints lie inside the prompt.
    let prompt = vec![user(copy(passage(1, 60)))];
    for layout in layouts() {
        let cold = rig(&path, layout, false, &|_, _| {}).ask(&prompt, 40);
        let cached = rig(&path, layout, true, &|_, _| {});
        let first = cached.ask(&prompt, 40);
        let second = cached.ask(&prompt, 40);
        report("first", layout, &first);
        report("second", layout, &second);
        assert_same("the first answer", &cold, &first);
        assert_same("the second answer", &cold, &second);
        assert_eq!(first.usage.cached_prompt_tokens, 0);
        let pages = second.usage.prompt_tokens / PAGE * PAGE;
        assert!(
            second.usage.cached_prompt_tokens >= pages,
            "{layout:?}: the full pages of the prompt come from the cache: {:?}",
            second.usage
        );
        let c = cached.snapshot().counters;
        eprintln!(
            "RES {layout:?}: {} checkpoints saved, {} restored, {} rows replayed, {} rows \
             computed in all, {} page copies",
            c.checkpoints_saved,
            c.checkpoints_restored,
            c.replay_rows,
            c.prefill_rows,
            c.page_copies
        );
        assert!(c.prefix_hits >= 1 && c.checkpoints_saved >= 1 && c.checkpoints_restored >= 1);
        assert!(c.page_copies >= 1, "the tail of the first run is copied before the second writes");
        assert!(c.replay_rows > 0 && c.replay_rows < u64::from(second.usage.prompt_tokens));
        assert!(cached.snapshot().checkpoint_ram > 0, "a checkpoint was kept in locked memory");
    }
}

#[test]
fn a_conversation_continued_over_several_turns_is_answered_as_without_a_cache() {
    let Some(path) = model_path() else { return };
    for layout in layouts() {
        let cold = rig(&path, layout, false, &|_, _| {});
        let cached = rig(&path, layout, true, &|_, _| {});
        let mut history = vec![user(copy(passage(2, 30)))];
        let mut before = 0;
        for turn in 0..5 {
            let want = cold.ask(&history, 48);
            let got = cached.ask(&history, 48);
            report(&format!("turn {turn}"), layout, &got);
            assert_same(&format!("turn {turn}"), &want, &got);
            if turn > 0 {
                // The prompt of the turn before is read from the cache, its
                // whole pages at least. (Not its answer: the chat template
                // renders an assistant message other than the model wrote it,
                // so the text parts from what is cached where the answer
                // begins.)
                assert!(
                    got.usage.cached_prompt_tokens >= before / PAGE * PAGE,
                    "{layout:?} turn {turn}: {:?} after a prompt of {before} tokens",
                    got.usage
                );
            }
            // The last tokens of a prompt, the start of the answer, are
            // rendered differently the next time.
            before = got.usage.prompt_tokens.saturating_sub(16);
            history.push(Message::new(Role::Assistant, got.text));
            history.push(user(copy(passage(10 + turn, 12))));
        }
        let c = cached.snapshot().counters;
        eprintln!(
            "RES {layout:?} conversation: {} hits, {} cells from the cache, {} restored, {} \
             replayed, {} computed",
            c.prefix_hits, c.cached_cells, c.checkpoints_restored, c.replay_rows, c.prefill_rows
        );
        assert!(c.prefix_hits >= 4 && c.checkpoints_restored >= 4);
    }
}

#[test]
fn a_prompt_that_parts_from_a_cached_one_inside_a_page_is_answered_as_without_a_cache() {
    let Some(path) = model_path() else { return };
    // The same 300 or so tokens, then two different questions: the second
    // prompt shares two pages and part of a third with the first.
    let common = passage(3, 18);
    let a = vec![user(format!("{common}\n\nCopy the first sentence of the text above."))];
    let b = vec![user(format!("{common}\n\nCopy the last sentence of the text above."))];
    for layout in layouts() {
        let cold = rig(&path, layout, false, &|_, _| {}).ask(&b, 40);
        let cached = rig(&path, layout, true, &|_, _| {});
        let _ = cached.ask(&a, 8);
        let got = cached.ask(&b, 40);
        report("parting prompt", layout, &got);
        assert_same("the prompt that parts from the cached one", &cold, &got);
        let c = cached.snapshot().counters;
        assert!(
            got.usage.cached_prompt_tokens > PAGE
                && !got.usage.cached_prompt_tokens.is_multiple_of(PAGE),
            "{layout:?}: the shared text ends inside a page: {:?}",
            got.usage
        );
        assert!(
            c.page_copies >= 1,
            "{layout:?}: the partly used page is copied before it is written"
        );
    }
}

#[test]
fn two_equal_prompts_at_once_are_answered_alike_and_the_second_reads_what_the_first_computes() {
    let Some(path) = model_path() else { return };
    let prompt = vec![user(copy(passage(4, 60)))];
    for layout in layouts() {
        let cold = rig(&path, layout, false, &no_drafts).ask(&prompt, 40);
        let cached = rig(&path, layout, true, &no_drafts);
        let first = cached.submit(&prompt, 40);
        let second = cached.submit(&prompt, 40);
        let (first, second) = (
            first.recv_timeout(Duration::from_secs(180)).unwrap(),
            second.recv_timeout(Duration::from_secs(180)).unwrap(),
        );
        report("first of two", layout, &first);
        report("second of two", layout, &second);
        assert_same("the first of two equal prompts", &cold, &first);
        assert_same("the second of two equal prompts", &cold, &second);
        let c = cached.snapshot().counters;
        eprintln!(
            "RES {layout:?}: {} requests held back, {} rows computed",
            c.requests_held, c.prefill_rows
        );
        assert!(c.requests_held >= 1, "{layout:?}: the second waited for the first's pages");
        let total = u64::from(first.usage.prompt_tokens);
        assert!(c.prefill_rows < 2 * total, "{layout:?}: the prompt was not computed twice");
    }
}

#[test]
fn requests_that_give_way_under_a_small_page_pool_come_back_from_the_cache() {
    let Some(path) = model_path() else { return };
    let prompts = [vec![user(copy(passage(5, 30)))], vec![user(copy(passage(6, 30)))]];
    for layout in layouts() {
        let cold = rig(&path, layout, false, &no_drafts);
        let want: Vec<Answer> = prompts.iter().map(|p| cold.ask(p, 160)).collect();
        // Two prompts of four pages and answers of two more do not fit a pool
        // of ten pages together, but the accounting lets both start: it
        // expects answers of 8 tokens. One gives way and comes back.
        let small = |config: &mut RuntimeConfig, _: u64| {
            config.speculation = false;
            config.pages = 10;
            config.output_clip = 8;
        };
        let cached = rig(&path, layout, true, &small);
        let pending: Vec<_> = prompts.iter().map(|p| cached.submit(p, 160)).collect();
        for (i, (want, got)) in want.iter().zip(pending).enumerate() {
            let got = got.recv_timeout(Duration::from_secs(180)).unwrap();
            report(&format!("request {i} under a small pool"), layout, &got);
            assert_same(&format!("request {i}"), want, &got);
        }
        let c = cached.snapshot().counters;
        eprintln!(
            "RES {layout:?}: {} requests gave way, {} hits, {} cells from the cache, {} \
             restored, {} replayed, {} pages evicted",
            c.requests_suspended,
            c.prefix_hits,
            c.cached_cells,
            c.checkpoints_restored,
            c.replay_rows,
            c.cache_pages_evicted
        );
        assert!(c.requests_suspended >= 1, "{layout:?}: nobody gave way; make the pool smaller");
        assert!(c.cached_cells > 0, "{layout:?}: whoever gave way came back from the cache");
        assert_eq!(c.cache_pages_evicted_late, 0);
    }
}

#[test]
fn a_small_checkpoint_budget_gives_up_checkpoints_and_the_answers_stay_the_same() {
    let Some(path) = model_path() else { return };
    for layout in layouts() {
        let cold = rig(&path, layout, false, &|_, _| {});
        // Room for two checkpoints, while a turn leaves several.
        let small = |config: &mut RuntimeConfig, checkpoint: u64| {
            config.checkpoint_budget = 2 * checkpoint;
        };
        let cached = rig(&path, layout, true, &small);
        let mut history = vec![user(copy(passage(7, 30)))];
        for turn in 0..5 {
            let want = cold.ask(&history, 48);
            let got = cached.ask(&history, 48);
            report(&format!("turn {turn} with two places"), layout, &got);
            assert_same(&format!("turn {turn}"), &want, &got);
            history.push(Message::new(Role::Assistant, got.text));
            history.push(user(copy(passage(20 + turn, 12))));
        }
        let c = cached.snapshot().counters;
        eprintln!(
            "RES {layout:?} two places: {} saved, {} restored, {} dropped, {} rows replayed",
            c.checkpoints_saved, c.checkpoints_restored, c.checkpoints_dropped, c.replay_rows
        );
        assert!(c.checkpoints_dropped >= 1, "{layout:?}: the budget forced a checkpoint out");
        assert!(c.checkpoints_restored >= 1, "{layout:?}: some checkpoint was still there");
        let held = cached.snapshot().checkpoint_ram;
        let places = 2 * cached.checkpoint;
        assert!(
            held > 0 && held <= places.next_multiple_of(32 << 20),
            "{layout:?}: {held} bytes locked for two places of {} bytes",
            cached.checkpoint
        );
    }
}

/// The tokens and logits of the picks `from..` of `rig`.
fn picks_since(rig: &Rig, from: usize) -> Vec<Pick> {
    rig.picks.lock().unwrap()[from..].to_vec()
}

/// Compares the picks of two runs of one request: the tokens must be equal;
/// returns how many logits are bit for bit the same and the widest difference
/// relative to the logit (at least 1).
fn compare_picks(what: &str, cold: &[Pick], cached: &[Pick]) -> (usize, f32) {
    assert_eq!(cold.len(), cached.len(), "{what}: one pick per generated token");
    let tokens = cold.iter().zip(cached).position(|(a, b)| a.token != b.token);
    assert_eq!(tokens, None, "{what}: the picks part at pick {tokens:?}");
    let same =
        cold.iter().zip(cached).filter(|(a, b)| a.logit.to_bits() == b.logit.to_bits()).count();
    let widest = cold
        .iter()
        .zip(cached)
        .map(|(a, b)| ((a.logit - b.logit) / a.logit.abs().max(1.0)).abs())
        .fold(0.0f32, f32::max);
    (same, widest)
}

#[test]
fn the_picks_of_a_request_that_starts_from_the_cache_have_the_logits_they_have_without_it() {
    let Some(path) = model_path() else { return };
    let prompt = vec![user(copy(passage(8, 60)))];
    for layout in layouts() {
        let cold = rig(&path, layout, false, &no_drafts);
        cold.ask(&prompt, 40);
        let want = picks_since(&cold, 0);
        let cached = rig(&path, layout, true, &no_drafts);
        cached.ask(&prompt, 40);
        let first = picks_since(&cached, 0);
        let before = first.len();
        let second = cached.ask(&prompt, 40);
        let got = picks_since(&cached, before);
        assert!(second.usage.cached_prompt_tokens > 800, "{:?}", second.usage);
        let (same_first, widest_first) = compare_picks("the first run", &want, &first);
        let (same, widest) = compare_picks("the run from the cache", &want, &got);
        eprintln!(
            "RES {layout:?}: {} picks per run; the run that computes everything: {same_first} \
             logits bit for bit those of the cold engine, widest difference {widest_first:.1e}; \
             the run from the cache: {same} bit for bit, widest difference {widest:.1e}",
            want.len()
        );
        // Rounding, not state: a state that was not restored changes the
        // first logit by a good part of itself.
        assert!(widest < 5e-3, "{layout:?}: the logits part by {widest:.1e}");
    }
}

#[test]
fn a_row_computed_in_a_step_of_another_size_differs_by_rounding_only() {
    // Why the answers above are not bit for bit the cold ones in general: the
    // rounding of a row depends on the size of the step that computes it (and
    // on where in it the row is). The cache changes the steps that compute a
    // prompt: rows replayed from a checkpoint, the last row alone. This reads
    // one prompt in steps of several sizes on one card and compares every
    // row's top logit with the steps of 128 rows.
    let Some(path) = model_path() else { return };
    let options = Options {
        precision: Precision::Exact,
        max_rows: 128,
        logit_rows: 128,
        window_rows: 4,
        ..Options::default()
    };
    let model = Arc::new(
        Qwen35::new(Gguf::open(&path).unwrap(), "q", &Spread::Single, options, 2048).unwrap(),
    );
    let text = passage(5, 30);
    let ids = model
        .tokenizer()
        .encode(&text, ares_tokenizer::EncodeOptions { add_special: true, parse_special: false });
    let device = Device::open(0, Wait::Block).unwrap();
    let mut session = Session::start(model, vec![device], 2, 1024, 256 << 20).unwrap();
    let mut run = |chunk: usize, slot: u32| -> Vec<Pick> {
        let mut out = Vec::new();
        for (i, rows) in ids.chunks(chunk).enumerate() {
            let part = Rows {
                slot,
                position: (i * chunk) as u32,
                tokens: rows,
                direct: true,
                wanted: Wanted::All,
            };
            out.extend(session.step(&[part]).unwrap().iter().copied());
        }
        out
    };
    let base = run(128, 0);
    for chunk in [32, 64, 1, 7, 100] {
        let got = run(chunk, 1);
        let different_logits =
            got.iter().zip(&base).filter(|(a, b)| a.logit.to_bits() != b.logit.to_bits()).count();
        let different_tokens = got.iter().zip(&base).filter(|(a, b)| a.token != b.token).count();
        let widest = got
            .iter()
            .zip(&base)
            .map(|(a, b)| ((a.logit - b.logit) / b.logit.abs().max(1.0)).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "RES steps of {chunk} rows against steps of 128, {} rows: {different_logits} with \
             another top logit, {different_tokens} with another top token, widest difference \
             {widest:.1e}",
            ids.len()
        );
        assert!(widest < 5e-3, "steps of {chunk} rows: {widest:.1e}");
        assert!(different_tokens * 100 <= ids.len(), "steps of {chunk} rows: {different_tokens}");
        if chunk % 32 == 0 {
            assert_eq!(different_logits, 0, "steps of whole tiles of 32 rows give the same rows");
        }
    }
}
