//! `qwen4exp` on devices against the CPU reference, on a made-up model with
//! every part of the architecture at sizes the kernels take.
//!
//! The tests open GPUs and are skipped unless `ARES_TEST_CUDA=1` is set.
//! `ARES_TEST_DEVICES` names the devices by the driver's index (`0`, or `0,1`
//! for two; default `0`). Two devices get the layers split between them;
//! with `ARES_TEST_SPREAD=tensor` both work on every layer instead
//! (`ARES_TEST_MIXER` names which of the two holds the streams, default the
//! first; `ARES_TEST_SHARED=mixer` puts the shared expert there). The model
//! then has 96 experts per layer instead of 48, so that each device owns as
//! many as one device holds of a layer in the other tests; `ARES_TEST_EXPERTS`
//! sets the number for any run.
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
use ares_cuda::{CacheFormat, Device, ExpertShape, Precision, Wait, device_count};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    LayerOut, Options, PHASES, Phase, Probe, Profile, Rows, Session, SessionOptions, Spread, Wanted,
};
use ares_model::qwen4exp::synthetic::{NgramShape, Shape, Synthetic, Types};
use ares_model::qwen4exp::{Config, LogitRows, Recorder, Reference, SequenceState, Weights};
use ares_runtime::expert_store::Serve;

/// The spread the tests of two devices ask for: both on every layer when
/// `ARES_TEST_SPREAD=tensor`.
fn tensor() -> Option<Spread> {
    let var = |name: &str| std::env::var(name).ok();
    (var("ARES_TEST_SPREAD").as_deref() == Some("tensor")
        && var("ARES_TEST_DEVICES").is_some_and(|d| d.contains(',')))
    .then(|| Spread::Tensor {
        mixer: Some(var("ARES_TEST_MIXER").and_then(|m| m.parse().ok()).unwrap_or(0)),
        core_share: Some(0.5),
        shared_on_mixer: Some(var("ARES_TEST_SHARED").as_deref() == Some("mixer")),
    })
}

/// The layers whose experts one device has slots for: all of them when both
/// devices work on every layer.
fn layers_per_device(n_layer: usize, devices: usize) -> usize {
    if tensor().is_some() { n_layer } else { n_layer / devices }
}

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
        // Twice the experts when two devices share them: each then owns as
        // many per layer as one device holds of a layer otherwise.
        // (`ARES_TEST_EXPERTS` gives a split by layers the same model, to
        // compare the two.)
        n_expert: std::env::var("ARES_TEST_EXPERTS")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(if tensor().is_some() { 96 } else { 48 }),
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
        spread: tensor().unwrap_or(Spread::Layers),
        bounds: (devices == 2 && tensor().is_none()).then(|| vec![0, n_layer / 2, n_layer]),
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

/// Whether `pick` is the most likely token of `want` up to what the measured `error` of the
/// logits (relative RMS) can move: a token within three times that of the largest logit,
/// and within a hundredth of the logits' RMS (the error is a mean over the row; the logits
/// of a made-up model's top tokens lie close together).
fn is_near_best(pick: u32, want: &[f32], error: f64) -> bool {
    let best = want.iter().copied().fold(f32::MIN, f32::max);
    let rms = (want.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    f64::from(best - want[pick as usize]) <= (3.0 * error + 1e-2) * rms
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

/// Every layer of steps of 9 to 256 rows, each fed the reference's input, against the
/// reference: with the dense matrices as f32 (exact products) and quantized (the tile
/// kernels' integer products), with half of the experts without a slot.
#[test]
fn every_layer_of_wide_steps_matches_the_reference() {
    let Some(devices) = devices() else { return };
    for (types, core_bound) in [(Types::Shipped, 2e-3), (Types::Quantized, 4e-2)] {
        let synthetic = Synthetic { types, ..model(11) };
        let gguf = open(
            &format!("wide-layers-{types:?}"),
            &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0),
        );
        let c = Config::from_gguf(&gguf).unwrap();
        let w = Weights::load(&gguf, &c).unwrap();
        let reference = Reference::new(&c, &w);
        let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
        opts.part.max_rows = 256;
        let mut session =
            Session::open(&gguf, &c, &w, devices.clone(), opts).expect("the model opens");
        let present = c.experts.count as u32 / 2;
        session.fill(&mut |_| (0..present).collect()).expect("the tiers fill");
        let steps = [9usize, 64, 5, 256, 1, 17];
        let all = synthetic.shape.tokens(steps.iter().sum(), 3);
        let mut state = SequenceState::new(&c);
        let mut worst = [[0.0f64; 3]; 2];
        let (mut rows, mut same_picks, mut at) = (0, 0, 0);
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
            "  {types:?}, {} device(s): recurrent layers: core {:.2e}, experts {:.2e}, residual \
             {:.2e}; attention layers: core {:.2e}, experts {:.2e}, residual {:.2e}; \
             {same_picks} of {rows} rows of layers with the reference's experts",
            devices.len(),
            worst[0][0],
            worst[0][1],
            worst[0][2],
            worst[1][0],
            worst[1][1],
            worst[1][2],
        );
        assert!(worst[0][0] < core_bound, "recurrent core {:.2e}", worst[0][0]);
        assert!(worst[1][0] < core_bound.max(3e-2), "attention core {:.2e}", worst[1][0]);
        // With quantized dense matrices the 8-bit activations of the products move the
        // router's input a little, and a row whose picks flip at a near tie gets other
        // experts: the error of the experts' output is that of those rows.
        let (experts, residual, picks) =
            if types == Types::Quantized { (0.3, 6e-2, 94) } else { (8e-2, 4e-2, 97) };
        for kind in &worst {
            assert!(kind[1] < experts, "experts {:.2e}", kind[1]);
            assert!(kind[2] < residual, "residual {:.2e}", kind[2]);
        }
        assert!(same_picks * 100 >= rows * picks, "{same_picks} of {rows} routings");
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

/// The slices of a prompt of 9, 64 and 256 rows run on the many-row kernels, with half of
/// every layer's experts without a slot, and what follows them (a step of a few rows and
/// single tokens) continues from the state they left. The dense products take int8
/// activations (`Precision::Fast` on these cards) or f16 ones (`Precision::Exact`); and
/// with so few slots that the misses of a layer come in several batches, each giving way
/// to the one after it.
#[test]
fn steps_wider_than_eight_rows_match_the_reference() {
    let Some(devices) = devices() else { return };
    for (precision, few_slots) in
        [(Precision::Fast, false), (Precision::Exact, false), (Precision::Exact, true)]
    {
        wide_steps(&devices, precision, few_slots);
    }
}

fn wide_steps(devices: &[Device], precision: Precision, few_slots: bool) {
    let synthetic = Synthetic { types: Types::Quantized, ..model(23) };
    let gguf = open("wide", &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
    opts.part.max_rows = 256;
    opts.part.precision = precision;
    if few_slots {
        // The least the store gives a layer: the experts of a step of eight rows.
        let shape = ExpertShape {
            n_embd: c.n_embd as u32,
            n_ff: c.experts.ff as u32,
            n_experts: c.experts.count as u32,
            top_k: c.experts.used as u32,
            gate_ty: GgmlType::IQ2_S,
            down_ty: GgmlType::Q2_0,
        };
        let layers = layers_per_device(c.n_layer, devices.len()) as u64;
        opts.slot_bytes = Some(layers * u64::from(8 * shape.top_k) * shape.slot_bytes().unwrap());
    }
    opts.part.cells = 512;
    let mut session =
        Session::open(&gguf, &c, &w, devices.to_vec(), opts).expect("the model opens");
    let present = if few_slots { 6 } else { c.experts.count as u32 / 2 };
    session.fill(&mut |_| (0..present).collect()).expect("the tiers fill");
    let mut state = SequenceState::new(&c);
    let prompt = synthetic.shape.tokens(9 + 64 + 256 + 3, 9);
    let (mut at, mut worst, mut token) = (0, 0.0f64, 0u32);
    for rows in [9, 64, 256, 3] {
        let tokens = &prompt[at..at + rows];
        at += rows;
        let picks = session
            .step(&[Rows { slot: 0, tokens, direct: true, wanted: Wanted::Last }])
            .unwrap()
            .to_vec();
        let window = reference.window(&state, tokens, LogitRows::All).unwrap();
        let want = window.logits_of(rows - 1).unwrap();
        let error = rel_rms(session.logits(0).unwrap(), want);
        eprintln!(
            "  {precision:?}{}, {rows} rows: logits of the last row differ by {error:.2e} (relative \
             RMS)",
            if few_slots { ", few slots" } else { "" }
        );
        assert!(is_near_best(picks[0], want, error), "the most likely token after {at} tokens");
        worst = worst.max(error);
        token = picks[0];
        reference.commit(&mut state, &window, rows).unwrap();
    }
    let wide = session.take_wide_stats();
    eprintln!(
        "  {} layers of wide steps: {} experts picked, {} of them without a slot, {} more rounds",
        wide.layers, wide.picked, wide.missing, wide.extra_rounds
    );
    assert!(wide.missing > 0, "no expert was missing: the test brings none");
    if few_slots {
        assert!(wide.extra_rounds > wide.layers, "the misses came in one batch: {wide:?}");
    }
    // The quantized dense matrices cost the 8-bit activations of their products.
    assert!(worst < 0.2, "logits {worst:.2e}");
    // Single tokens go on from the state the wide steps left.
    for _ in 0..8 {
        let picks = session
            .step(&[Rows { slot: 0, tokens: &[token], direct: true, wanted: Wanted::Last }])
            .unwrap()
            .to_vec();
        let window = reference.window(&state, &[token], LogitRows::Last).unwrap();
        let want = window.logits_of(0).unwrap();
        let error = rel_rms(session.logits(0).unwrap(), want);
        assert!(is_near_best(picks[0], want, error), "a single token after the wide steps");
        reference.commit(&mut state, &window, 1).unwrap();
        token = arg_max(want);
    }
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
    let layers_per_device = layers_per_device(c.n_layer, devices.len());
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

/// What every layer of a step gave: the residual after it, the output of its
/// recurrent or attention part, the output of its experts.
#[derive(Default)]
struct Kept {
    layers: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

impl Probe for Kept {
    fn before(&mut self, _layer: usize) -> Option<Vec<f32>> {
        None
    }

    fn after(&mut self, _layer: usize, out: LayerOut<'_>) {
        self.layers.push((out.residual.to_vec(), out.core.to_vec(), out.ffn.to_vec()));
    }
}

/// Feeds every layer the residual another session had before it and keeps,
/// per row, the largest difference of what the layer gives from what it gave
/// there: of the experts' output, of the core's, of the residual.
struct Fed<'a> {
    /// What the first layer reads, and what the other session's layers gave.
    first: &'a [f32],
    theirs: &'a Kept,
    n: usize,
    wide: usize,
    worst: [f64; 3],
}

impl Probe for Fed<'_> {
    fn before(&mut self, layer: usize) -> Option<Vec<f32>> {
        Some(match layer {
            0 => self.first.to_vec(),
            _ => self.theirs.layers[layer - 1].0.clone(),
        })
    }

    fn after(&mut self, layer: usize, out: LayerOut<'_>) {
        let (residual, core, ffn) = &self.theirs.layers[layer];
        let rows = |ours: &[f32], theirs: &[f32], width: usize| -> f64 {
            ours.chunks(width)
                .zip(theirs.chunks(width))
                .map(|(o, t)| rel_rms(o, t))
                .fold(0.0, f64::max)
        };
        let errors = [
            rows(out.ffn, ffn, self.n),
            rows(out.core, core, self.n),
            rows(out.residual, residual, self.wide),
        ];
        for (worst, error) in self.worst.iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
    }
}

/// Runs a text on two sessions, one with every expert in its slot and one
/// where half of every layer's experts have a slot and misses are served as
/// `serve` says, each layer of the second reading the residual the first had
/// before that layer. Returns the largest relative RMS difference of a row
/// of a layer between the two: of the experts' output, of the core's output
/// and of the residual after the layer.
fn layers_with_misses(devices: &[Device], serve: Serve) -> [f64; 3] {
    let synthetic = model(31);
    let name = format!("fed-{serve:?}");
    let gguf = open(&name, &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let text = synthetic.shape.tokens(56, 29);
    let mut whole = session(&gguf, &c, &w, devices.to_vec(), CacheFormat::Q8);

    let slots = c.experts.used * 8;
    let shape = ExpertShape {
        n_embd: c.n_embd as u32,
        n_ff: c.experts.ff as u32,
        n_experts: c.experts.count as u32,
        top_k: c.experts.used as u32,
        gate_ty: GgmlType::IQ2_S,
        down_ty: GgmlType::Q2_0,
    };
    let layers_per_device = layers_per_device(c.n_layer, devices.len());
    let opts = SessionOptions {
        slot_bytes: Some((slots * layers_per_device) as u64 * shape.slot_bytes().unwrap()),
        serve,
        ..options(devices.len(), c.n_layer, CacheFormat::Q8)
    };
    let mut missing = Session::open(&gguf, &c, &w, devices.to_vec(), opts).unwrap();
    let n_expert = c.experts.count as u32;
    missing.fill(&mut |_| (0..n_expert).collect()).unwrap();
    missing.take_store_stats();

    let (n, wide) = (c.n_embd, c.wide());
    let mut worst = [0.0f64; 3];
    for tokens in text[..40].chunks(8).chain(text[40..].chunks(1)) {
        let rows = [Rows { slot: 0, tokens, direct: true, wanted: Wanted::None }];
        let mut theirs = Kept::default();
        whole.step_probed(&rows, &mut theirs).unwrap();
        // The first layer reads the embeddings, in every stream.
        let mut first = vec![0.0f32; tokens.len() * wide];
        for (row, &token) in tokens.iter().enumerate() {
            for stream in first[row * wide..(row + 1) * wide].chunks_exact_mut(n) {
                w.token_embd.decode_row(token as usize, stream);
            }
        }
        let mut fed = Fed { first: &first, theirs: &theirs, n, wide, worst };
        missing.step_probed(&rows, &mut fed).unwrap();
        worst = fed.worst;
    }
    let stats = missing.take_store_stats();
    let misses = stats.picks - stats.hits;
    eprintln!(
        "  {} device(s), {serve:?}: {misses} of {} picks had no slot, {} of them computed on the \
         CPU ({} uploads, {} promotions between steps); fed the residuals of a run without \
         misses, a layer's rows differ from that run's by at most {:.2e} (experts), {:.2e} \
         (core), {:.2e} (residual), relative RMS",
        devices.len(),
        stats.picks,
        stats.cpu_experts,
        stats.uploads,
        stats.promotions,
        worst[0],
        worst[1],
        worst[2],
    );
    assert!(misses > 100, "the run was to miss experts: {misses}");
    match serve {
        Serve::Stream => assert_eq!(stats.cpu_experts, 0),
        Serve::Cpu => assert_eq!(stats.cpu_experts, misses),
        Serve::Measured => {}
    }
    worst
}

/// An expert that is in no slot may be computed on the CPU instead, in the
/// kernels' own arithmetic with another order of the float additions: every
/// layer then gives what it gives with every expert in its slot, to the
/// seventh digit (and to a rounding step of one hidden value where a value
/// sits on the boundary between two 8-bit codes).
///
/// The layers are compared one by one, each fed the residual of the run
/// without misses, because two free runs of this model cannot be: its 8-bit
/// activation codes turn a difference in the seventh digit into one in the
/// third within a few layers, and its discrete choices turn that into
/// another answer where two candidates are close. Row 45 of this text is
/// such a place (the two blocks its last attention layer chooses between
/// score 43.09 and 42.95), and of free runs with misses on the CPU one in
/// four gave the other answer there, 0.115 away, as a run with every expert
/// in its slot does when its residual is disturbed by a hundredth. That is
/// the model, not the engine: `experts_uploaded_for_a_step_change_nothing`
/// can compare free runs only because uploads change no bit at all.
#[test]
fn experts_computed_on_the_cpu_change_nothing() {
    let Some(devices) = devices() else { return };
    for serve in [Serve::Cpu, Serve::Measured] {
        let [experts, core, residual] = layers_with_misses(&devices, serve);
        // The core reads the same residual and the same state: no bit differs.
        assert_eq!(core, 0.0, "{serve:?}");
        // An expert left out or added twice is some tenths of a layer's rows.
        assert!(experts < 1e-2, "{serve:?}: experts {experts:.2e}");
        assert!(residual < 1e-2, "{serve:?}: residual {residual:.2e}");
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
    let slots = (c.experts.used * 8 * layers_per_device(c.n_layer, devices.len())) as u64;
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
        // A worker has no mixers: it exchanges with the device that has.
        let mixes = ms[Phase::Mixers as usize] > 0.0;
        let works = tensor().is_some() && ms[Phase::Exchange as usize] > 0.0;
        assert!(ms[Phase::Experts as usize] > 0.0 && (mixes || works));
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
    let slots = (c.experts.used * 8 * layers_per_device(c.n_layer, devices.len())) as u64;
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
