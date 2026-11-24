//! Images through the conductor, without a GPU: a model that reads images
//! and an executor that plays its tower and checks every step's rows.
//!
//! The executor's tower gives row `j` of an image the values `(seed, j, nx,
//! ny)`, so every row a step stages says which row of which image it is. The
//! executor checks that the rows of every image cell are staged, in order, and
//! that every row has the rotary position llama.cpp's `mtmd` gives it: text one
//! after the other, an image's rows `(p, p + j / nx, p + j % nx)`, the text
//! after an image `max(nx, ny)` on. It can also play a model that pauses a long step of one
//! prompt between its pieces (a prompt read layer by layer): every record that goes on with
//! the step must stage the same rows again.

mod common;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{ChatFormat, EngineError, GenerateRequest, ModelCard, Prompt, Sampling};
use ares_runtime::testing::Pauses;
use ares_runtime::{
    Clock, ContextSpec, DeviceContext, DeviceModel, EncodedPrompt, ExecError, Executor, ImageInput,
    ImageSpan, LinkSpec, Model, ModelError, Pick, RecordView, Runtime, RuntimeConfig, Split,
    StepMeta, StepOutput, TowerJob, Vision, WakeCounts, Woke,
};
use ares_state::PageLayout;
use common::*;

const WIDTH: u32 = 4;
const UNITS: u32 = 5;
const START: u32 = 900;
const PAD: u32 = 901;
const END: u32 = 902;

/// A model of words and images: `w` is a token, `[s nx ny]` an image of seed
/// `s` with `nx x ny` rows between its two markers.
struct Pictures;

impl Model for Pictures {
    fn card(&self) -> ModelCard {
        ModelCard { id: "pictures".into(), context_length: 4096, created: 0 }
    }

    fn vocab(&self) -> u32 {
        1000
    }

    fn ends(&self, _token: u32) -> bool {
        false
    }

    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        Ok(self.encode_with_boundaries(prompt)?.tokens)
    }

    fn encode_with_boundaries(&self, prompt: &Prompt) -> Result<EncodedPrompt, EngineError> {
        let Prompt::Text(text) = prompt else { unreachable!("text prompts only") };
        let mut out = EncodedPrompt::default();
        for word in text.split(' ') {
            if let Some(image) = word.strip_prefix('[') {
                let n: Vec<u32> =
                    image.trim_end_matches(']').split(',').map(|n| n.parse().unwrap()).collect();
                let (seed, nx, ny) = (n[0], n[1], n[2]);
                out.tokens.push(START);
                out.images.push(ImageSpan {
                    first: out.tokens.len() as u32,
                    image: Arc::new(ImageInput {
                        name: [seed as u8; 32],
                        width: nx * 32,
                        height: ny * 32,
                        pixels: Vec::new(),
                        nx,
                        ny,
                    }),
                });
                out.tokens.extend(std::iter::repeat_n(PAD, (nx * ny) as usize));
                out.tokens.push(END);
            } else {
                out.tokens.push(5);
            }
        }
        Ok(out)
    }

    fn chat_format(&self) -> Option<ChatFormat> {
        None
    }

    fn piece(&self, _token: u32, text: &mut Vec<u8>) {
        text.push(b'x');
    }

    fn context(&self) -> ContextSpec {
        ContextSpec::NONE
    }

    fn split(&self) -> Split {
        Split::Single
    }

    fn devices(&self) -> usize {
        1
    }

    fn links(&self) -> LinkSpec {
        LinkSpec::default()
    }

    fn max_rows(&self) -> u32 {
        64
    }

    fn prompt_rows(&self) -> u32 {
        8
    }

    fn logit_rows(&self) -> u32 {
        8
    }

    fn draft_rows(&self) -> u32 {
        0
    }

    fn vision(&self) -> Option<Vision> {
        Some(Vision { width: WIDTH, max_rows: 64, device: 0, units: UNITS })
    }

    fn page_entry(&self, page: u32) -> u32 {
        page
    }

    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        PageLayout::new(1, cells_per_page)
    }

    fn open(&self, _cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        Err(ModelError::Invalid("a model for the host executor of this test".into()))
    }
}

/// What the executor saw.
#[derive(Default, Debug)]
struct Seen {
    complaints: Vec<String>,
    /// Image rows staged and checked.
    image_rows: u64,
    /// Passes of the tower that ended, and the jobs it took.
    passes: u32,
    jobs: u32,
    /// Steps paused between their pieces.
    paused: u32,
}

/// Per slot: the rotary position the next cell must have, and where.
#[derive(Clone, Copy, Debug)]
struct Next {
    cell: u32,
    rotary: u32,
}

struct Checker {
    record: Vec<u8>,
    stage: Vec<f32>,
    used: usize,
    clock: Clock,
    next: HashMap<u32, Next>,
    /// The pass under way: the image's seed and the next unit.
    pass: Option<(u8, u32)>,
    /// The steps of one prompt that run in pieces, each paused but the last, and the time a
    /// piece takes.
    pauses: Option<Pauses>,
    piece: Duration,
    seen: Arc<Mutex<Seen>>,
}

impl Checker {
    fn new(stage_rows: usize, seen: Arc<Mutex<Seen>>) -> Self {
        Self {
            record: vec![0; 1 << 20],
            stage: vec![0.0; stage_rows * WIDTH as usize],
            used: 0,
            clock: Clock::new(),
            next: HashMap::new(),
            pass: None,
            pauses: None,
            piece: Duration::ZERO,
            seen,
        }
    }
}

impl Executor for Checker {
    fn devices(&self) -> usize {
        1
    }

    fn clock(&self) -> Clock {
        self.clock
    }

    fn record(&mut self) -> &mut [u8] {
        &mut self.record
    }

    fn image_stage(&mut self) -> &mut [f32] {
        &mut self.stage
    }

    fn tower(&mut self, job: &mut TowerJob) -> Result<(), ExecError> {
        let mut seen = self.seen.lock().unwrap();
        seen.jobs += 1;
        let seed = job.image.name[0];
        let expected = match self.pass {
            Some((s, next)) if s == seed => next,
            Some((s, _)) => {
                seen.complaints.push(format!("image {seed} began while image {s} was in its pass"));
                0
            }
            None => 0,
        };
        if job.units.start != expected || job.units.end <= job.units.start {
            seen.complaints.push(format!("units {:?} after unit {expected}", job.units));
        }
        if job.units.end < UNITS {
            self.pass = Some((seed, job.units.end));
            return Ok(());
        }
        self.pass = None;
        seen.passes += 1;
        let (nx, ny) = (job.image.nx, job.image.ny);
        job.rows =
            (0..nx * ny).flat_map(|j| [f32::from(seed), j as f32, nx as f32, ny as f32]).collect();
        job.ms = 1.0;
        Ok(())
    }

    fn dispatch(&mut self, meta: &StepMeta) -> Result<(), ExecError> {
        self.used = meta.record_bytes;
        Ok(())
    }

    fn collect(&mut self, _spin: Duration, out: &mut StepOutput) -> Result<Woke, ExecError> {
        let view = RecordView::new(&self.record[..self.used]).unwrap();
        std::thread::sleep(self.piece);
        // A step of one prompt wider than the pauses' rows runs in pieces.
        out.paused = match &mut self.pauses {
            Some(pauses) => pauses.step(&view).is_some_and(|done| done < pauses.pieces()),
            None => false,
        };
        // The rows are checked in every piece; only the last one moves the positions on.
        let kept = self.next.clone();
        let mut seen = self.seen.lock().unwrap();
        // Which rows take staged rows.
        let mut staged: HashMap<u32, [f32; 4]> = HashMap::new();
        for i in 0..view.image_runs() {
            let run = view.image_run(i);
            for k in 0..run.rows {
                let at = ((run.stage + k) * WIDTH) as usize;
                let row = &self.stage[at..at + 4];
                staged.insert(run.row + k, [row[0], row[1], row[2], row[3]]);
            }
        }
        for s in 0..view.segments() {
            let segment = view.segment(s);
            for k in 0..segment.n_rows {
                let i = segment.row0 + k;
                let (row, rotary) = (view.row(i), view.rotary(i));
                let next = self.next.get(&row.slot).copied();
                let expect = match next {
                    Some(next) if next.cell == row.position => Some(next.rotary),
                    _ if row.position == 0 => Some(0),
                    _ => None,
                };
                match (row.token, staged.get(&i)) {
                    (PAD, Some(&[_, j, nx, ny])) => {
                        let (j, nx, ny) = (j as u32, nx as u32, ny as u32);
                        // The image's rotary position, from its first row on.
                        let base = if j == 0 { expect } else { Some(rotary[0]) };
                        if let Some(p) = base
                            && rotary != [p, p + j / nx, p + j % nx, 0]
                        {
                            seen.complaints.push(format!(
                                "row {j} of an image of {nx} x {ny} at cell {} is at {rotary:?} \
                                 with the image at {p}",
                                row.position
                            ));
                        }
                        seen.image_rows += 1;
                        let after =
                            if j + 1 == nx * ny { rotary[0] + nx.max(ny) } else { rotary[0] };
                        self.next.insert(row.slot, Next { cell: row.position + 1, rotary: after });
                    }
                    (PAD, None) => seen
                        .complaints
                        .push(format!("the image cell {} has no staged row", row.position)),
                    (_, Some(_)) => seen
                        .complaints
                        .push(format!("the text cell {} has a staged row", row.position)),
                    (_, None) => {
                        let p = rotary[0];
                        if rotary != [p, p, p, 0] || expect.is_some_and(|e| e != p) {
                            seen.complaints.push(format!(
                                "the text cell {} is at {rotary:?}, after {next:?}",
                                row.position
                            ));
                        }
                        self.next.insert(row.slot, Next { cell: row.position + 1, rotary: p + 1 });
                    }
                }
            }
        }
        out.picks.clear();
        if out.paused {
            self.next = kept;
            seen.paused += 1;
        } else {
            out.picks.extend((0..view.picked_rows()).map(|_| Pick {
                token: 7,
                logit: 0.0,
                aux: 0.0,
            }));
        }
        out.logprobs.clear();
        out.drafts.clear();
        out.draft_probs.clear();
        out.inexact.clear();
        out.devices.clear();
        out.step_ms = Some(1.0);
        Ok(Woke::Spinning)
    }

    fn device_wakes(&self) -> WakeCounts {
        WakeCounts::default()
    }

    fn shutdown(&mut self) {}
}

fn runtime(stage_rows: usize) -> (Runtime, Arc<Mutex<Seen>>) {
    runtime_with(stage_rows, |config| config, |checker| checker)
}

/// [`runtime`] with the config and the executor changed.
fn runtime_with(
    stage_rows: usize,
    config: impl FnOnce(RuntimeConfig) -> RuntimeConfig,
    checker: impl FnOnce(Checker) -> Checker,
) -> (Runtime, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let config = config(RuntimeConfig {
        pages: 256,
        cells_per_page: 16,
        max_slots: 4,
        window_rows: 1,
        warm_up: false,
        ..RuntimeConfig::default()
    });
    let executor = checker(Checker::new(stage_rows, seen.clone()));
    (Runtime::start(Arc::new(Pictures), Box::new(executor), config).unwrap(), seen)
}

fn ask(runtime: &Runtime, prompt: &str, tokens: u32) -> Run {
    let request = GenerateRequest {
        model: "pictures".into(),
        prompt: Prompt::Text(prompt.into()),
        sampling: Sampling::default(),
        max_tokens: Some(tokens),
        ..common::request("", None)
    };
    run_one(runtime, request)
}

#[test]
fn image_rows_are_staged_at_their_cells_and_every_row_has_its_rotary_position() {
    // A stage of 5 rows: a piece reads at most that many image rows, so the
    // images of 12 and 6 rows are read in parts.
    let (runtime, seen) = runtime(5);
    let a = "w w [1,4,3] w w w";
    let done = ask(&runtime, a, 6);
    assert_eq!(done.usage.completion_tokens, 6);
    // The same image again, after text of another length, and a second one,
    // tall: its rows come from the store, the new one's from the tower.
    let b = "w [1,4,3] w w [2,2,3] w";
    let done = ask(&runtime, b, 4);
    assert_eq!(done.usage.prompt_tokens, 1 + 14 + 2 + 8 + 1);
    let seen = seen.lock().unwrap();
    assert!(seen.complaints.is_empty(), "{:#?}", seen.complaints);
    assert_eq!(seen.image_rows, 12 + 12 + 6);
    assert_eq!(seen.passes, 2, "the first image's rows were kept");
    drop(seen);
    runtime.shutdown();
}

#[test]
fn prompts_with_images_next_to_each_other_each_get_their_own_rows() {
    let (runtime, seen) = runtime(64);
    let handles: Vec<_> = (0..3u32)
        .map(|i| {
            let prompt = format!("w [{},{},{}] w w", 10 + i, 2 + i, 3);
            let request = GenerateRequest {
                model: "pictures".into(),
                prompt: Prompt::Text(prompt),
                sampling: Sampling::default(),
                max_tokens: Some(5),
                ..common::request("", None)
            };
            start(&runtime, request).unwrap()
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.finish().usage.completion_tokens, 5);
    }
    let seen = seen.lock().unwrap();
    assert!(seen.complaints.is_empty(), "{:#?}", seen.complaints);
    assert_eq!(seen.image_rows, 6 + 9 + 12);
    assert_eq!(seen.passes, 3);
    drop(seen);
    runtime.shutdown();
}

/// A step of a prompt with an image that the model pauses between its pieces stages the
/// image's rows again in every record that goes on with it, also when the image store let
/// them go meanwhile for another request's image (the store holds one image here).
#[test]
fn a_paused_step_stages_its_image_rows_again_after_the_store_let_them_go() {
    let (runtime, seen) = runtime_with(
        64,
        // One image of 12 rows of 4 values.
        |config| RuntimeConfig { image_rows: 12 * 16, ..config },
        |mut checker| {
            checker.pauses = Some(Pauses::new(20, 40));
            checker.piece = Duration::from_millis(2);
            checker
        },
    );
    // The first image's rows come into the store with a short prompt; the long one then
    // reads them in its first step, which pauses.
    assert_eq!(ask(&runtime, "w [1,4,3] w", 2).usage.completion_tokens, 2);
    let words = |n: usize| vec!["w"; n].join(" ");
    let long = format!("{} [1,4,3] {}", words(10), words(30));
    let request = |prompt: String, tokens: u32| GenerateRequest {
        model: "pictures".into(),
        prompt: Prompt::Text(prompt),
        sampling: Sampling::default(),
        max_tokens: Some(tokens),
        ..common::request("", None)
    };
    let long_run = start(&runtime, request(long, 4)).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while seen.lock().unwrap().paused < 1 {
        assert!(std::time::Instant::now() < deadline, "the step never paused");
        std::thread::sleep(Duration::from_micros(200));
    }
    // Another image while the long prompt's step is paused: the store keeps it in place of
    // the first.
    let short = start(&runtime, request("w [2,2,3] w".into(), 3)).unwrap();
    assert_eq!(short.finish().usage.completion_tokens, 3);
    assert_eq!(long_run.finish().usage.completion_tokens, 4);
    let seen = seen.lock().unwrap();
    assert!(seen.complaints.is_empty(), "{:#?}", seen.complaints);
    assert!(seen.paused >= 2, "{} pauses", seen.paused);
    assert_eq!(seen.passes, 2, "the paused step's rows were held for it, not computed again");
    drop(seen);
    runtime.shutdown();
}
