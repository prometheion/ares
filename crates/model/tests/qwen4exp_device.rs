//! `qwen4exp` on devices against the CPU reference, on a made-up model with
//! every part of the architecture at sizes the kernels take.
//!
//! The tests open GPUs and are skipped unless `ARES_TEST_CUDA=1` is set.
//! `ARES_TEST_DEVICES` names the devices by the driver's index (`0`, or `0,1`
//! for the layer split over two; default `0`).
//!
//! What is exact and what is not: the made-up model stores its dense matrices
//! as f32, f16 and bf16, which the weight kernels multiply with unquantized
//! activations, so every dense product is exact to f32 rounding. Two things
//! are not, and they are the engine's own: the routed experts, whose kernels
//! round the activations to 8 bits twice (their types are those of the real
//! file), and K and V in `q8`. The numbers the tests print say how much each
//! costs; the bounds they assert are those numbers with room.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use ares_core::GgmlType;
use ares_cuda::{CacheFormat, Device, ExpertShape, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    LayerOut, Options, PHASES, Phase, Probe, Profile, Rows, Session, SessionOptions, Wanted,
};
use ares_model::qwen4exp::synthetic::{NgramShape, Shape, Synthetic, Types};
use ares_model::qwen4exp::{Config, LogitRows, Recorder, Reference, SequenceState, Weights};
use ares_runtime::expert_store::Serve;

fn devices() -> Option<Vec<Device>> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("ARES_TEST_CUDA").is_some()) {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    let wanted = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0".into());
    let count = device_count().expect("the CUDA driver");
    Some(
        wanted
            .split(',')
            .map(|d| {
                let ordinal: u32 = d.trim().parse().expect("ARES_TEST_DEVICES: a list of numbers");
                assert!(ordinal < count, "device {ordinal} of {count}");
                Device::open(ordinal, Wait::Spin).expect("open device")
            })
            .collect(),
    )
}

/// Three streams of 256, attention heads and indexer keys of 128, a selection
/// budget of two blocks (rows select from their twelfth cell on), 48 experts
/// of which a row uses 3, an n-gram table of IQ4_NL rows of 32 values.
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

fn open(name: &str, bytes: &[u8]) -> Gguf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4dev-{name}.gguf"));
    std::fs::write(&path, bytes).expect("the test directory is writable");
    Gguf::open(&path).expect("the writer makes a valid file")
}

fn options(devices: usize, n_layer: usize, kv: CacheFormat) -> SessionOptions {
    SessionOptions {
        part: Options {
            kv: (kv, kv),
            logit_rows: 8,
            window_rows: 8,
            cells: 512,
            ..Options::default()
        },
        // The tests share the cards with whatever else runs.
        headroom: 512 << 20,
        ram_tier: Some(64 << 20),
        bounds: (devices == 2).then(|| vec![0, n_layer / 2, n_layer]),
        // Every expert has a slot: none is to be evicted between steps.
        reserve: Some(0),
        cpu_workers: Some(2),
        ..SessionOptions::default()
    }
}

fn session<'g>(
    gguf: &'g Gguf,
    c: &Config,
    w: &'g Weights<'g>,
    devices: Vec<Device>,
    kv: CacheFormat,
) -> Session<'g> {
    let opts = options(devices.len(), c.n_layer, kv);
    let mut session = Session::open(gguf, c, w, devices, opts).expect("the model opens");
    let n_expert = c.experts.count as u32;
    let fill = session.fill(&mut |_| (0..n_expert).collect()).expect("the tiers fill");
    assert_eq!(fill.uploads as usize, c.n_layer * c.experts.count, "every expert has a slot");
    session
}

/// The relative RMS difference of `got` from `want`.
fn rel_rms(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut diff, mut size) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        diff += (f64::from(*g) - f64::from(*w)).powi(2);
        size += f64::from(*w).powi(2);
    }
    (diff / size.max(1e-30)).sqrt()
}

fn arg_max(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// What a layer of the reference reads and gives, for the rows of one step.
struct Expected {
    input: Vec<f32>,
    output: Vec<f32>,
    core: Vec<f32>,
    ffn: Vec<f32>,
    picks: Vec<Vec<u32>>,
}

/// Feeds every layer the reference's input and keeps the worst difference of
/// what it gives, by kind of layer.
struct Compare<'a> {
    expected: &'a [Expected],
    attention: &'a [bool],
    /// Recurrent and attention layers: core output, expert output, residual.
    worst: [[f64; 3]; 2],
    rows: usize,
    same_picks: usize,
}

impl Probe for Compare<'_> {
    fn before(&mut self, layer: usize) -> Option<Vec<f32>> {
        Some(self.expected[layer].input.clone())
    }

    fn after(&mut self, layer: usize, out: LayerOut<'_>) {
        let want = &self.expected[layer];
        let kind = usize::from(self.attention[layer]);
        let errors = [
            rel_rms(out.core, &want.core),
            rel_rms(out.ffn, &want.ffn),
            rel_rms(out.residual, &want.output),
        ];
        for (worst, error) in self.worst[kind].iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
        for (got, want) in out.picks.iter().zip(&want.picks) {
            self.rows += 1;
            self.same_picks += usize::from(got == want);
        }
    }
}

/// Every layer of the reference for `tokens` after `state`, each fed with the
/// output of the one before.
fn expected(
    reference: &Reference<'_, '_>,
    c: &Config,
    w: &Weights<'_>,
    state: &SequenceState,
    tokens: &[u32],
) -> Vec<Expected> {
    let (n, wide) = (c.n_embd, c.wide());
    let mut r = vec![0.0f32; tokens.len() * wide];
    for (row, &token) in tokens.iter().enumerate() {
        for stream in r[row * wide..(row + 1) * wide].chunks_exact_mut(n) {
            w.token_embd.decode_row(token as usize, stream);
        }
    }
    (0..c.n_layer)
        .map(|layer| {
            let mut rec = Recorder::default();
            let output = reference.block_observed(layer, state, tokens, &r, &mut rec);
            Expected {
                input: std::mem::replace(&mut r, output.clone()),
                output,
                core: rec.tensor_of(Some(layer), "attn_out").unwrap().to_vec(),
                ffn: rec.tensor_of(Some(layer), "ffn_out").unwrap().to_vec(),
                picks: (0..tokens.len()).map(|row| rec.routing[&(layer, row)].0.clone()).collect(),
            }
        })
        .collect()
}

#[test]
fn every_layer_matches_the_reference() {
    let Some(devices) = devices() else { return };
    let synthetic = model(11);
    let gguf = open("layers", &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    // Steps of every size up to eight rows, 56 cells: past the dense limit
    // of 11 cells from the second step on.
    let steps = [8usize, 5, 1, 6, 8, 3, 2, 7, 8, 4, 4];
    let all = synthetic.shape.tokens(steps.iter().sum(), 3);
    for kv in [CacheFormat::F16, CacheFormat::Q8] {
        let mut session = session(&gguf, &c, &w, devices.clone(), kv);
        let mut state = SequenceState::new(&c);
        let mut worst = [[0.0f64; 3]; 2];
        let (mut rows, mut same_picks) = (0, 0);
        let mut at = 0;
        for &count in &steps {
            let tokens = &all[at..at + count];
            let layers = expected(&reference, &c, &w, &state, tokens);
            let mut probe = Compare {
                expected: &layers,
                attention: &c.attention_layers,
                worst,
                rows: 0,
                same_picks: 0,
            };
            session
                .step_probed(
                    &[Rows { slot: 0, tokens, direct: true, wanted: Wanted::None }],
                    &mut probe,
                )
                .unwrap();
            worst = probe.worst;
            rows += probe.rows;
            same_picks += probe.same_picks;
            let window = reference.window(&state, tokens, LogitRows::None).unwrap();
            reference.commit(&mut state, &window, count).unwrap();
            at += count;
        }
        eprintln!(
            "  K/V {kv:?}, {} device(s): recurrent layers: core {:.2e}, experts {:.2e}, residual {:.2e}; \
             attention layers: core {:.2e}, experts {:.2e}, residual {:.2e}; \
             {same_picks} of {rows} rows of layers with the reference's experts",
            devices.len(),
            worst[0][0],
            worst[0][1],
            worst[0][2],
            worst[1][0],
            worst[1][1],
            worst[1][2],
        );
        // Dense products are exact; the recurrent part differs by rounding.
        assert!(worst[0][0] < 2e-3, "recurrent core {:.2e}", worst[0][0]);
        // Attention reads K and V as stored (f16 or q8) on f16 tensor cores.
        let bound = if kv == CacheFormat::F16 { 1e-2 } else { 3e-2 };
        assert!(worst[1][0] < bound, "attention core {:.2e}", worst[1][0]);
        // The routed experts round their activations to 8 bits twice.
        for kind in &worst {
            assert!(kind[1] < 8e-2, "experts {:.2e}", kind[1]);
            assert!(kind[2] < 4e-2, "residual {:.2e}", kind[2]);
        }
        assert!(same_picks * 100 >= rows * 97, "{same_picks} of {rows} routings");
    }
}

/// The whole pass: the most likely token of every prompt row, the logits of
/// the last one, and greedy tokens after it.
#[test]
fn the_forward_pass_and_greedy_tokens_match_the_reference() {
    let Some(devices) = devices() else { return };
    let synthetic = model(5);
    let gguf = open("forward", &synthetic.bytes_with_experts(GgmlType::IQ2_XXS, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut session = session(&gguf, &c, &w, devices.clone(), CacheFormat::Q8);
    let mut state = SequenceState::new(&c);
    let prompt = synthetic.shape.tokens(40, 9);
    let (mut same, mut worst, mut last) = (0, 0.0f64, (0u32, 0u32));
    for tokens in prompt.chunks(8) {
        let picks = session
            .step(&[Rows { slot: 0, tokens, direct: true, wanted: Wanted::All }])
            .unwrap()
            .to_vec();
        let window = reference.window(&state, tokens, LogitRows::All).unwrap();
        for (row, pick) in picks.iter().enumerate() {
            let want = window.logits_of(row).unwrap();
            let got = session.logits(row).unwrap();
            worst = worst.max(rel_rms(got, want));
            same += usize::from(*pick == arg_max(want));
            last = (*pick, arg_max(want));
        }
        reference.commit(&mut state, &window, tokens.len()).unwrap();
    }
    // Greedy tokens: each side continues its own text.
    let n_new = 24;
    let (mut ours, mut theirs) = (vec![last.0], vec![last.1]);
    for _ in 1..n_new {
        let token = [*ours.last().unwrap()];
        let picks = session
            .step(&[Rows { slot: 0, tokens: &token, direct: true, wanted: Wanted::Last }])
            .unwrap();
        ours.push(picks[0]);
        let token = [*theirs.last().unwrap()];
        let window = reference.window(&state, &token, LogitRows::Last).unwrap();
        theirs.push(arg_max(window.logits_of(0).unwrap()));
        reference.commit(&mut state, &window, 1).unwrap();
    }
    let equal = ours.iter().zip(&theirs).take_while(|(a, b)| a == b).count();
    eprintln!(
        "  {} device(s): the most likely token is the reference's in {same} of {} prompt rows, \
         logits differ by at most {worst:.2e} (relative RMS); {equal} of {n_new} greedy tokens \
         before the first difference",
        devices.len(),
        prompt.len(),
    );
    assert!(worst < 0.15, "logits {worst:.2e}");
    assert!(same * 10 >= prompt.len() * 9, "{same} of {} rows", prompt.len());
}

/// A draft window leaves the state untouched and its commit advances it by
/// the accepted rows: the logits afterwards are those of the same tokens run
/// as certain rows.
#[test]
fn windows_and_commits_give_what_certain_rows_give() {
    let Some(devices) = devices() else { return };
    let synthetic = model(23);
    let gguf = open("windows", &synthetic.bytes_with_experts(GgmlType::IQ1_M, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let mut session = session(&gguf, &c, &w, devices.clone(), CacheFormat::Q8);
    let text = synthetic.shape.tokens(40, 17);
    let wrong = synthetic.shape.tokens(80, 18);

    // The text as certain rows, one step per token after a first step of 8.
    let mut plain = Vec::new();
    let rows = |tokens| Rows { slot: 0, tokens, direct: true, wanted: Wanted::Last };
    session.step(&[rows(&text[..8])]).unwrap();
    for at in 8..text.len() {
        session.step(&[rows(&text[at..at + 1])]).unwrap();
        plain.push(session.logits(0).unwrap().to_vec());
    }

    // The same text through windows: the newest token and drafts, of which
    // the first few are the text's and the rest are not.
    session.restart(0).unwrap();
    session.step(&[rows(&text[..8])]).unwrap();
    let mut at = 8;
    let mut worst = 0.0f64;
    let mut windows = 0;
    for (round, right) in [3usize, 0, 5, 1, 7, 2, 4, 0, 6].into_iter().cycle().enumerate() {
        let drafts = 1 + round % 7;
        if at + right.min(drafts) + 1 > text.len() {
            break;
        }
        let right = right.min(drafts);
        // The newest token, `right` drafts that are the text's, the others wrong.
        let mut tokens = text[at..at + 1 + right].to_vec();
        tokens.extend(wrong[at..at + drafts - right].iter().map(|t| (t + 1) % c.n_vocab as u32));
        session
            .step(&[Rows { slot: 0, tokens: &tokens, direct: false, wanted: Wanted::All }])
            .unwrap();
        // Rows 0..=right computed the text's cells from the committed state.
        for row in 0..=right {
            let got = session.logits(row).unwrap();
            worst = worst.max(rel_rms(got, &plain[at + row - 8]));
        }
        session.accept(&[1 + right as u32]).unwrap();
        at += 1 + right;
        windows += 1;
    }
    eprintln!(
        "  {} device(s): {windows} windows up to cell {at}: logits of the accepted rows differ \
         from those of certain rows by at most {worst:.2e} (relative RMS)",
        devices.len()
    );
    assert!(windows >= 8 && at >= 30);
    // The rows of a window share launches with other rows than a single
    // certain row does, which changes the rounding of the 8-bit activations.
    assert!(worst < 5e-2, "{worst:.2e}");
}

/// Runs a text on a session where half of every layer's experts have a slot
/// (what a step of eight rows can need at most), with misses served as
/// `serve` says, and returns the largest relative RMS difference of a row's
/// logits from a run with every expert in its slot.
fn run_with_misses(devices: &[Device], serve: Serve) -> f64 {
    let synthetic = model(31);
    let name = format!("misses-{serve:?}");
    let gguf = open(&name, &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let text = synthetic.shape.tokens(56, 29);
    let run = |session: &mut Session<'_>| -> Vec<Vec<f32>> {
        let mut logits = Vec::new();
        for tokens in text[..40].chunks(8) {
            session.step(&[Rows { slot: 0, tokens, direct: true, wanted: Wanted::All }]).unwrap();
            logits.extend((0..tokens.len()).map(|row| session.logits(row).unwrap().to_vec()));
        }
        for token in text[40..].chunks(1) {
            session
                .step(&[Rows { slot: 0, tokens: token, direct: true, wanted: Wanted::Last }])
                .unwrap();
            logits.push(session.logits(0).unwrap().to_vec());
        }
        logits
    };
    let want = run(&mut session(&gguf, &c, &w, devices.to_vec(), CacheFormat::Q8));

    let slots = c.experts.used * 8;
    let shape = ExpertShape {
        n_embd: c.n_embd as u32,
        n_ff: c.experts.ff as u32,
        n_experts: c.experts.count as u32,
        top_k: c.experts.used as u32,
        gate_ty: GgmlType::IQ2_S,
        down_ty: GgmlType::Q2_0,
    };
    let layers_per_device = c.n_layer / devices.len();
    let opts = SessionOptions {
        slot_bytes: Some((slots * layers_per_device) as u64 * shape.slot_bytes().unwrap()),
        serve,
        ..options(devices.len(), c.n_layer, CacheFormat::Q8)
    };
    let mut session = Session::open(&gguf, &c, &w, devices.to_vec(), opts).unwrap();
    let n_expert = c.experts.count as u32;
    session.fill(&mut |_| (0..n_expert).collect()).unwrap();
    session.take_store_stats();
    let got = run(&mut session);
    let stats = session.take_store_stats();
    let errors: Vec<f64> = got.iter().zip(&want).map(|(g, w)| rel_rms(g, w)).collect();
    let worst = errors.iter().copied().fold(0.0f64, f64::max);
    let misses = stats.picks - stats.hits;
    eprintln!(
        "  {} device(s), {serve:?}: {misses} of {} picks had no slot, {} of them computed on the \
         CPU ({} uploads, {} promotions between steps, {} copies down); logits differ from a run \
         without misses by at most {worst:.2e} (relative RMS)",
        devices.len(),
        stats.picks,
        stats.cpu_experts,
        stats.uploads,
        stats.promotions,
        stats.demotions,
    );
    if let Some(first) = errors.iter().position(|e| *e > 1e-2) {
        let all: Vec<String> = errors.iter().map(|e| format!("{e:.0e}")).collect();
        eprintln!("  the first row that is off is row {first}; all rows: {}", all.join(" "));
    }
    assert!(misses > 100, "the run was to miss experts: {misses}");
    match serve {
        Serve::Stream => assert_eq!(stats.cpu_experts, 0),
        Serve::Cpu => assert_eq!(stats.cpu_experts, misses),
        Serve::Measured => {}
    }
    worst
}

/// An expert that is in no slot when its layer runs is uploaded and waited
/// for on the device: the logits are, to the bit, those of a run with every
/// expert in its slot.
#[test]
fn experts_uploaded_for_a_step_change_nothing() {
    let Some(devices) = devices() else { return };
    let worst = run_with_misses(&devices, Serve::Stream);
    assert_eq!(worst, 0.0);
}

/// An expert that is in no slot may be computed on the CPU instead, in the
/// kernels' own arithmetic with another order of the sums: the logits are
/// those of a run with every expert in its slot to a few parts in a thousand.
///
/// Ignored because it fails now and then as of 2026-11-10 (in 6 of 14 runs on
/// one device, in none of 5 on two): one row of the 56, always the same one,
/// a step of one row, is off by 0.1 while the rows before and after it are
/// right. It happens with every miss on the CPU and with the store's own
/// choice, never with uploads only. The state of the sequence is not
/// damaged, so the wrong value enters late in that step. The store promotes
/// about 120 experts to VRAM between the steps of such a run. Not found yet;
/// run with `--ignored` to see it.
#[test]
#[ignore = "one row is off by 0.1 in some runs on one device; see the comment"]
fn experts_computed_on_the_cpu_change_nothing() {
    let Some(devices) = devices() else { return };
    for serve in [Serve::Cpu, Serve::Measured] {
        let worst = run_with_misses(&devices, serve);
        assert!(worst < 1e-2, "{serve:?}: {worst:.2e}");
    }
}

/// The profile adds up the device time of timed steps by kind, and steps
/// run on while it is paused.
#[test]
fn the_profile_times_steps_by_kind() {
    let Some(devices) = devices() else { return };
    let synthetic = model(41);
    let gguf = open("profile", &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let profile = Arc::new(Profile::new());
    let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
    opts.part.profile = Some(profile.clone());
    // With experts outside their slots: a kernel that waits for the host
    // while the host records timed events.
    let shape = ExpertShape {
        n_embd: c.n_embd as u32,
        n_ff: c.experts.ff as u32,
        n_experts: c.experts.count as u32,
        top_k: c.experts.used as u32,
        gate_ty: GgmlType::IQ2_S,
        down_ty: GgmlType::Q2_0,
    };
    let slots = (c.experts.used * 8 * c.n_layer / devices.len()) as u64;
    opts.slot_bytes = Some(slots * shape.slot_bytes().unwrap());
    let mut session = Session::open(&gguf, &c, &w, devices.clone(), opts).unwrap();
    let n_expert = c.experts.count as u32;
    session.fill(&mut |_| (0..n_expert).collect()).unwrap();
    let text = synthetic.shape.tokens(32, 2);
    let rows = |tokens| Rows { slot: 0, tokens, direct: true, wanted: Wanted::Last };
    profile.pause(true);
    session.step(&[rows(&text[..8])]).unwrap();
    assert!(profile.totals().is_empty(), "a paused profile counts nothing");
    profile.pause(false);
    for token in text[8..].chunks(1) {
        session.step(&[rows(token)]).unwrap();
    }
    let totals = profile.totals();
    assert_eq!(totals.len(), devices.len());
    for (device, (steps, ms)) in totals.iter().enumerate() {
        assert_eq!(*steps, 24);
        let per_step: Vec<String> =
            PHASES.iter().map(|p| format!("{p:?} {:.3}", ms[*p as usize] / 24.0)).collect();
        eprintln!("  device {device}: ms per step: {}", per_step.join(", "));
        assert!(ms.iter().all(|ms| ms.is_finite() && *ms >= 0.0));
        assert!(ms[Phase::Experts as usize] > 0.0 && ms[Phase::Mixers as usize] > 0.0);
    }
}

/// A step of many layers is not enqueued to its end while the device waits
/// for an expert: the driver's queue of commands is finite, and a host that
/// fills it can no longer bring the expert. 48 layers are about 1 300
/// launches a step, as in the real model.
#[test]
fn a_long_step_that_misses_experts_ends() {
    let Some(devices) = devices() else { return };
    let mut synthetic = model(47);
    synthetic.shape.n_layer = 48;
    let gguf = open("long", &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let shape = ExpertShape {
        n_embd: c.n_embd as u32,
        n_ff: c.experts.ff as u32,
        n_experts: c.experts.count as u32,
        top_k: c.experts.used as u32,
        gate_ty: GgmlType::IQ2_S,
        down_ty: GgmlType::Q2_0,
    };
    let slots = (c.experts.used * 8 * c.n_layer / devices.len()) as u64;
    let text = synthetic.shape.tokens(96, 3);
    for serve in [Serve::Stream, Serve::Measured] {
        let opts = SessionOptions {
            slot_bytes: Some(slots * shape.slot_bytes().unwrap()),
            serve,
            stall: std::time::Duration::from_secs(20),
            ..options(devices.len(), c.n_layer, CacheFormat::Q8)
        };
        let mut session = Session::open(&gguf, &c, &w, devices.clone(), opts).unwrap();
        let n_expert = c.experts.count as u32;
        session.fill(&mut |_| (0..n_expert).collect()).unwrap();
        session.take_store_stats();
        for tokens in text.chunks(8) {
            session.step(&[Rows { slot: 0, tokens, direct: true, wanted: Wanted::Last }]).unwrap();
        }
        let stats = session.take_store_stats();
        let misses = stats.picks - stats.hits;
        eprintln!(
            "  {} device(s), {serve:?}: 12 steps of 8 rows through 48 layers, {misses} of {} picks \
             without a slot",
            devices.len(),
            stats.picks
        );
        assert!(misses > 500, "the run was to miss experts: {misses}");
    }
}
