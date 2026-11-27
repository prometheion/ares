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
use std::sync::Arc;

use ares_core::GgmlType;
use ares_cuda::{
    CacheFormat, Device, ExpertShape, Precision, Prune, PruneRule, PruneWeights, peer_access,
};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    LayerOut, ModelOptions, Options, PHASES, Phase, Probe, Profile, PruneSetup, RoleMemory, Rows,
    Session, SessionOptions, Spread, Wanted,
};
use ares_model::qwen4exp::synthetic::{NgramShape, Shape, Synthetic, Types};
use ares_model::qwen4exp::{Config, LogitRows, Recorder, Reference, SequenceState, Weights};
use ares_runtime::expert_store::{ExpertNeighbours, Near, ProfileKey, Serve, StoreStats};
use common::gpu::devices;
use common::helpers::is_near_best;

mod common;

/// The spread the tests of two devices ask for: both on every layer when
/// `ARES_TEST_SPREAD=tensor`. `ARES_TEST_CORE_SHARE` is the mixer's share of
/// the recurrent heads (0.5 unless given): with 1 and `ARES_TEST_SHARED=mixer`
/// the worker holds experts and nothing else.
fn tensor() -> Option<Spread> {
    let var = |name: &str| std::env::var(name).ok();
    (var("ARES_TEST_SPREAD").as_deref() == Some("tensor")
        && var("ARES_TEST_DEVICES").is_some_and(|d| d.contains(',')))
    .then(|| Spread::Tensor {
        mixer: Some(var("ARES_TEST_MIXER").and_then(|m| m.parse().ok()).unwrap_or(0)),
        core_share: Some(var("ARES_TEST_CORE_SHARE").and_then(|m| m.parse().ok()).unwrap_or(0.5)),
        shared_on_mixer: Some(var("ARES_TEST_SHARED").as_deref() == Some("mixer")),
    })
}

/// The layers whose experts one device has slots for: all of them when both
/// devices work on every layer.
fn layers_per_device(n_layer: usize, devices: usize) -> usize {
    if tensor().is_some() { n_layer } else { n_layer / devices }
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

/// [`model`] with experts whose down matrices `down_ty` can hold: an inner width of 64, or
/// of the type's block where that is larger.
fn model_of(seed: u64, down_ty: GgmlType) -> Synthetic {
    let mut synthetic = model(seed);
    synthetic.shape.expert_ff = (down_ty.block_values() as usize).max(64);
    synthetic
}

/// Every weight type once as the type of gate and up and once as the type of down: the
/// pair `i` has gate and up of type `i` and down of the type seven places on, so that the
/// pairs mix the families (k-quants with i-quants, floats with quantized types).
fn type_pairs() -> Vec<(GgmlType, GgmlType)> {
    let types: Vec<GgmlType> = ExpertShape::types().collect();
    (0..types.len()).map(|i| (types[i], types[(i + 7) % types.len()])).collect()
}

fn open(name: &str, bytes: &[u8]) -> Gguf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4dev-{name}.gguf"));
    std::fs::write(&path, bytes).expect("the test directory is writable");
    Gguf::open(&path).expect("the writer makes a valid file")
}

fn options(devices: usize, n_layer: usize, kv: CacheFormat) -> SessionOptions {
    let model = ModelOptions {
        part: Options {
            kv: (kv, kv),
            logit_rows: 8,
            window_rows: 8,
            cells: 512,
            ..Options::default()
        },
        // The tests share the cards with whatever else runs.
        headroom: 512 << 20,
        spread: tensor().unwrap_or(Spread::Layers),
        bounds: (devices == 2 && tensor().is_none()).then(|| vec![0, n_layer / 2, n_layer]),
        // Every expert has a slot: none is to be evicted between steps.
        reserve: Some(0),
        cpu_workers: Some(2),
        ..ModelOptions::default()
    };
    SessionOptions { model, ram_tier: Some(64 << 20), ..SessionOptions::default() }
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
    // Whether the room or the fill was short.
    let slots: usize = {
        let store = session.store();
        (0..store.n_layers()).map(|layer| store.slots(layer).0).sum()
    };
    assert_eq!(
        fill.uploads as usize,
        c.n_layer * c.experts.count,
        "every expert has a slot: {slots} slots in VRAM, {fill:?}"
    );
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
            // A maximum drops what is not a number.
            assert!(error.is_finite(), "layer {layer}: an output that is not a number");
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
    // With `q5` and `q4` the 56 cells are all in exact entries (the first tile
    // and the newest cells stay f16): the exact pool, its table and the exact
    // lists of the rows past the dense limit.
    for kv in [CacheFormat::F16, CacheFormat::Q8, CacheFormat::Q5, CacheFormat::Q4] {
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
        // Attention reads K and V as stored (f16 or q8; the exact cells of the
        // tile formats are f16) on f16 tensor cores.
        let bound = if kv == CacheFormat::Q8 { 3e-2 } else { 1e-2 };
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
/// reference: with the dense matrices as f32 (the float tile kernels: f16 operands, f32
/// sums) and quantized (the tile kernels' integer products), with half of the experts
/// without a slot.
///
/// Measured 2026-11-12 on the RTX 4070, before the float matrices of a wide step took the
/// tile kernels (they were multiplied exactly, eight rows at a time) and after. Float dense
/// matrices: recurrent core 3.6e-7 and 7.4e-4, experts 1.62e-2 and 1.65e-2, attention core
/// 7.9e-3 and 7.7e-3, 2814 and 2804 of 2816 rows with the reference's experts. Quantized:
/// recurrent core 1.57e-2 and 1.59e-2, experts 0.216 and 0.216, attention core 1.65e-2 and
/// 1.84e-2, 2734 and 2746 rows. With f16 sums instead of f32 sums for the float matrices
/// (what the step's precision would pick on these cards) the attention core of the
/// quantized model was 6.2e-2 and the experts of the float one 5.3e-2, which is why the
/// float matrices sum in f32.
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
        opts.model.part.max_rows = 256;
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

/// Two devices that can write each other's memory and share every layer time the second
/// device's role at load when the spread names none (`roles.rs`): computing a share of the
/// recurrent heads and the shared expert, and holding experts only. Each is timed in a
/// decode step and a wide step, the faster of the two together is kept and placed, and the
/// model placed that way reads a prompt as the reference does. (Two devices only; the
/// spread of the other tests is ignored.)
#[test]
fn the_second_devices_role_is_timed_at_load() {
    let Some(devices) = devices() else { return };
    let peers = devices.len() == 2
        && peer_access(devices[0].ordinal(), devices[1].ordinal()).unwrap().0
        && peer_access(devices[1].ordinal(), devices[0].ordinal()).unwrap().0;
    if !peers {
        eprintln!("skipped: two devices that can write each other's memory");
        return;
    }
    let synthetic = model(5);
    let gguf = open("roles", &synthetic.bytes_with_experts(GgmlType::IQ2_XXS, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut opts = options(2, c.n_layer, CacheFormat::Q8);
    opts.model.spread = Spread::Tensor { mixer: None, core_share: None, shared_on_mixer: None };
    opts.model.bounds = None;
    opts.model.part.max_rows = 64;
    let mut session = Session::open(&gguf, &c, &w, devices, opts).expect("the model opens");
    let tensor = session.placement().tensor.clone().expect("both devices on every layer");
    let trial = tensor.trial.clone().expect("the roles were timed");
    for time in &trial.times {
        eprintln!(
            "  core share {:.3}, the shared expert on the {}: decode {:.3} ms, a step of {} rows \
             {:.2} ms",
            time.choice.core_share,
            if time.choice.shared_on_mixer { "mixer" } else { "worker" },
            time.decode.as_secs_f64() * 1e3,
            trial.wide_rows,
            time.wide.unwrap_or_default().as_secs_f64() * 1e3
        );
    }
    eprintln!(
        "  kept core share {:.3}, trial {:.2} s",
        trial.kept.core_share,
        trial.took.as_secs_f64()
    );
    assert_eq!(trial.times.len(), 2, "both roles fit the cards: {trial:?}");
    assert_eq!(trial.wide_rows, 64);
    assert!(trial.times.iter().all(|t| t.wide.is_some()), "{trial:?}");
    assert!(trial.times[0].choice.core_share < 1.0 && !trial.times[0].choice.shared_on_mixer);
    assert!(trial.times[1].choice.core_share == 1.0 && trial.times[1].choice.shared_on_mixer);
    let fastest = trial.times.iter().min_by(|a, b| a.cost().total_cmp(&b.cost())).unwrap();
    assert_eq!(trial.kept, fastest.choice);
    assert_eq!(tensor.choice, trial.kept);
    // The model as placed: a prompt in slices of the decode kernels and a wide step.
    let n_expert = c.experts.count as u32;
    session.fill(&mut |_| (0..n_expert).collect()).expect("the tiers fill");
    let mut state = SequenceState::new(&c);
    let prompt = synthetic.shape.tokens(8 + 40, 9);
    let mut worst = 0.0f64;
    for tokens in [&prompt[..8], &prompt[8..]] {
        let picks = session
            .step(&[Rows { slot: 0, tokens, direct: true, wanted: Wanted::Last }])
            .unwrap()
            .to_vec();
        let window = reference.window(&state, tokens, LogitRows::All).unwrap();
        let want = window.logits_of(tokens.len() - 1).unwrap();
        let error = rel_rms(session.logits(0).unwrap(), want);
        assert!(is_near_best(picks[0], want, error), "the most likely token after the slice");
        worst = worst.max(error);
        reference.commit(&mut state, &window, tokens.len()).unwrap();
    }
    eprintln!("  the logits of the last rows differ by at most {worst:.2e} (relative RMS)");
    assert!(worst < 0.15, "logits {worst:.2e}");
}

/// With a directory to keep it in, the first load times the roles and writes what it
/// kept; the next load of the same model on the same cards takes it from there and
/// times nothing; a file that is not whole is measured again. (Two devices only.)
#[test]
fn the_second_devices_role_is_remembered_between_loads() {
    let Some(devices) = devices() else { return };
    let peers = devices.len() == 2
        && peer_access(devices[0].ordinal(), devices[1].ordinal()).unwrap().0
        && peer_access(devices[1].ordinal(), devices[0].ordinal()).unwrap().0;
    if !peers {
        eprintln!("skipped: two devices that can write each other's memory");
        return;
    }
    let synthetic = model(5);
    let gguf = open("roles-kept", &synthetic.bytes_with_experts(GgmlType::IQ2_XXS, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let dir = common::TempDir::new("roles");
    let mut opts = options(2, c.n_layer, CacheFormat::Q8);
    opts.model.spread = Spread::Tensor { mixer: None, core_share: None, shared_on_mixer: None };
    opts.model.bounds = None;
    opts.model.part.max_rows = 64;
    opts.model.roles_dir = Some(dir.path().to_owned());
    let trial_of = |opts: &SessionOptions| {
        let session = Session::open(&gguf, &c, &w, devices.clone(), opts.clone()).unwrap();
        let tensor = session.placement().tensor.clone().expect("both devices on every layer");
        assert_eq!(Some(tensor.choice), tensor.trial.as_ref().map(|t| t.kept));
        tensor.trial.expect("the roles were timed or remembered")
    };
    let first = trial_of(&opts);
    let RoleMemory::Written(path) = &first.memory else { panic!("{:?}", first.memory) };
    assert_eq!(path.extension().and_then(|e| e.to_str()), Some("roles"));
    let second = trial_of(&opts);
    assert_eq!(second.memory, RoleMemory::Remembered(path.clone()));
    assert_eq!((second.kept, &second.times), (first.kept, &first.times));
    std::fs::write(path, "ares roles 1\nkept 0\n").unwrap();
    let third = trial_of(&opts);
    assert_eq!(third.memory, RoleMemory::Written(path.clone()), "measured again");
    // Without a directory it is timed at every load.
    opts.model.roles_dir = None;
    assert_eq!(trial_of(&opts).memory, RoleMemory::None);
}

/// The slices of a prompt of 9, 64 and 256 rows run on the many-row kernels, with half of
/// every layer's experts without a slot, and what follows them (a step of a few rows and
/// single tokens) continues from the state they left. The dense products take int8
/// activations (`Precision::Fast` on these cards) or f16 ones (`Precision::Exact`). The
/// experts a layer misses pass through transit: with room for all of them at once, with
/// four transit slots (several rounds a layer, from RAM and from the file, and what the
/// heat ranks high takes a slot of the tier afterwards), and, with a tier that holds only
/// what a decode step needs, without transit: then they come into the layer's slots in
/// batches, each giving way to the one after it.
#[test]
fn steps_wider_than_eight_rows_match_the_reference() {
    let Some(devices) = devices() else { return };
    for (precision, slots) in [
        (Precision::Fast, Slots::Plenty),
        (Precision::Exact, Slots::Plenty),
        (Precision::Exact, Slots::Transit),
        (Precision::Exact, Slots::Few),
    ] {
        wide_steps(&devices, precision, slots);
    }
}

/// The slots a store of the wide steps' test has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slots {
    /// What the placement gives, and half of the experts in them.
    Plenty,
    /// What a decode step needs in every layer and four transit slots.
    Transit,
    /// What a decode step needs in every layer and nothing else: no transit.
    Few,
}

fn wide_steps(devices: &[Device], precision: Precision, slots: Slots) {
    let few_slots = slots != Slots::Plenty;
    let synthetic = Synthetic { types: Types::Quantized, ..model(23) };
    let gguf = open("wide", &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
    opts.model.part.max_rows = 256;
    opts.model.part.precision = precision;
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
        let transit = if slots == Slots::Transit { 4 } else { 0 };
        opts.model.slot_bytes =
            Some((layers * u64::from(8 * shape.top_k) + transit) * shape.slot_bytes().unwrap());
        opts.model.transit = Some(transit as u32);
    }
    opts.model.part.cells = 512;
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
            "  {precision:?}, {slots:?}, {rows} rows: logits of the last row differ by {error:.2e} \
             (relative RMS)"
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
    let stats = session.take_store_stats();
    eprintln!(
        "  transit: {} experts ({} from RAM, {} read from the file, {} read ahead), {} took a \
         slot afterwards, {} stayed in RAM; into slots: {} uploads, {} evictions",
        stats.transit_uploads,
        stats.transit_from_ram,
        stats.transit_reads,
        stats.transit_ahead_reads,
        stats.transit_adopted,
        stats.transit_admitted,
        stats.uploads,
        stats.evictions,
    );
    match slots {
        Slots::Few => assert_eq!(stats.transit_uploads, 0, "a tier at its floor has no transit"),
        Slots::Plenty => assert!(stats.transit_uploads > 0, "{stats:?}"),
        Slots::Transit => {
            assert!(stats.transit_from_ram > 0 && stats.transit_reads > 0, "{stats:?}");
            assert!(stats.transit_adopted > 0 && stats.evictions > 0, "{stats:?}");
        }
    }
    // The quantized dense matrices cost the 8-bit activations of their products, and the
    // rounding of a step is not the same from run to run: the many-row kernels add a row's
    // experts in the order they finish, and the misses come in rounds as transit brings them.
    // The 8-bit activation codes of the Fast precision make a difference of rounding one of
    // tenths in this made-up model (2026-11-27, RTX 4070, six runs: the last row after the
    // 256-row step 0.133 to 0.215 in Fast, at most 0.049 in Exact; the first step of 9 rows
    // 3.8e-2 or 5.3e-2 in Fast, 3.8e-3 or 2.4e-2 in Exact with transit). Each pick is a near
    // tie of the reference's at the measured error (above).
    let bound = if precision == Precision::Fast { 0.3 } else { 0.1 };
    assert!(worst < bound, "logits {worst:.2e}");
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
    run_with_misses_tuned(devices, serve, "", |_, _| {}).0
}

/// [`run_with_misses`] with the options of the run with misses as `tune` makes them (from
/// the model's configuration), and what its store counted.
fn run_with_misses_tuned(
    devices: &[Device],
    serve: Serve,
    tag: &str,
    tune: impl Fn(&Config, &mut SessionOptions),
) -> (f64, StoreStats) {
    let synthetic = model(31);
    let name = format!("misses-{serve:?}{tag}");
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
    let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
    opts.model.slot_bytes = Some((slots * layers_per_device) as u64 * shape.slot_bytes().unwrap());
    opts.model.serve = serve;
    tune(&c, &mut opts);
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
    (worst, stats)
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

/// The routers of the next layers predict the picks of a step of eight rows and the store
/// reads ahead what no tier holds where such reads are picked more often than not: the
/// logits are, to the bit, those of a run with every expert in its slot, with the
/// predictions and without them. (The made-up routers rarely pick what is in no tier, so
/// little is read here; the store's own tests read ahead, and the real file's fingerprint
/// compares with reads ahead.)
#[test]
fn experts_read_ahead_of_their_router_change_nothing() {
    let Some(devices) = devices() else { return };
    let small_ram = |c: &Config, opts: &mut SessionOptions| {
        let per_layer = if tensor().is_some() { devices.len() } else { 1 };
        opts.ram_tier = Some((4 * c.n_layer * per_layer) as u64 * (64 << 10));
    };
    for predict in [0, 1, 2] {
        let (worst, stats) =
            run_with_misses_tuned(&devices, Serve::Stream, &format!("-ahead{predict}"), |c, o| {
                small_ram(c, o);
                o.model.part.predict = predict;
            });
        eprintln!(
            "  predictions of {predict} layers ahead: {} named, {} of them picked; {} read \
             ahead, {} used, {} wasted, {} left; {} demand reads",
            stats.predicted,
            stats.predicted_picked,
            stats.ahead_reads,
            stats.ahead_used,
            stats.ahead_wasted,
            stats.ahead_left,
            stats.demand_reads,
        );
        assert_eq!(worst, 0.0);
        if predict == 0 {
            assert_eq!(stats.predicted, 0);
        } else {
            assert!(stats.predicted > 0 && stats.predicted_picked > 0, "{stats:?}");
        }
    }
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
                .map(|(o, t)| {
                    let error = rel_rms(o, t);
                    // A maximum drops what is not a number.
                    assert!(error.is_finite(), "layer {layer}: a row that is not a number");
                    error
                })
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
    layers_with_misses_of(devices, serve, GgmlType::IQ2_S, GgmlType::Q2_0)
}

/// [`layers_with_misses`] for a model whose experts have gate and up of `gate_ty` and down
/// of `down_ty`.
fn layers_with_misses_of(
    devices: &[Device],
    serve: Serve,
    gate_ty: GgmlType,
    down_ty: GgmlType,
) -> [f64; 3] {
    let synthetic = model_of(31, down_ty);
    let name = format!("fed-{serve:?}-{gate_ty}-{down_ty}");
    let gguf = open(&name, &synthetic.bytes_with_experts(gate_ty, down_ty));
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
        gate_ty,
        down_ty,
    };
    let layers_per_device = layers_per_device(c.n_layer, devices.len());
    let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
    opts.model.slot_bytes = Some((slots * layers_per_device) as u64 * shape.slot_bytes().unwrap());
    opts.model.serve = serve;
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
        "  {gate_ty} and {down_ty}, {} device(s), {serve:?}: {misses} of {} picks had no slot, {} \
         of them computed on the CPU ({} uploads, {} promotions between steps); fed the residuals \
         of a run without \
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

/// The misses of a wide step on the CPU (`hybrid/wide.rs` of the store): the experts few
/// rows of a slice of a prompt picked are computed from their RAM slots in the decode
/// steps' arithmetic (8-bit activations) instead of brought through transit and computed in
/// f16 on the device. Every layer, fed the residual of a run with every expert in its slot
/// (in wide steps too), gives what that run gives within the rounding of the 8-bit
/// activations (measured: 1.9e-2 for the experts, 4.5e-3 for the residual, with every miss
/// on the CPU; the many-row kernels' experts are 1.65e-2 from the reference's,
/// `every_layer_of_wide_steps_matches_the_reference`); the core reads the same residual
/// and gives the same bits.
#[test]
fn experts_a_wide_step_misses_computed_on_the_cpu_differ_by_their_rounding() {
    let Some(devices) = devices() else { return };
    let synthetic = model_of(31, GgmlType::Q2_0);
    let (gate_ty, down_ty) = (GgmlType::IQ2_S, GgmlType::Q2_0);
    let gguf = open("fed-wide", &synthetic.bytes_with_experts(gate_ty, down_ty));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let text = synthetic.shape.tokens(56, 29);
    let wide_rows = |mut opts: SessionOptions| {
        opts.model.part.max_rows = 64;
        opts
    };
    let n_expert = c.experts.count as u32;
    let shape = ExpertShape {
        n_embd: c.n_embd as u32,
        n_ff: c.experts.ff as u32,
        n_experts: c.experts.count as u32,
        top_k: c.experts.used as u32,
        gate_ty,
        down_ty,
    };
    let slots = c.experts.used * 8;
    let layers_per_device = layers_per_device(c.n_layer, devices.len());
    for serve in [Serve::Cpu, Serve::Measured] {
        // A run with every expert in its slot, in wide steps too, from the start of the text.
        let mut whole = Session::open(
            &gguf,
            &c,
            &w,
            devices.clone(),
            wide_rows(options(devices.len(), c.n_layer, CacheFormat::Q8)),
        )
        .unwrap();
        whole.fill(&mut |_| (0..n_expert).collect()).unwrap();
        let mut opts = wide_rows(options(devices.len(), c.n_layer, CacheFormat::Q8));
        opts.model.slot_bytes =
            Some((slots * layers_per_device) as u64 * shape.slot_bytes().unwrap());
        opts.model.serve = serve;
        let mut missing = Session::open(&gguf, &c, &w, devices.to_vec(), opts).unwrap();
        missing.fill(&mut |_| (0..n_expert).collect()).unwrap();
        missing.take_store_stats();
        let (n, wide) = (c.n_embd, c.wide());
        let mut worst = [0.0f64; 3];
        // Two wide steps of 25 rows, then one of six (the decode kernels).
        for tokens in text[..50].chunks(25).chain(text[50..].chunks(6)) {
            let rows = [Rows { slot: 0, tokens, direct: true, wanted: Wanted::None }];
            let mut theirs = Kept::default();
            whole.step_probed(&rows, &mut theirs).unwrap();
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
        let wide_stats = missing.take_wide_stats();
        eprintln!(
            "  {} device(s), {serve:?}: {} of {} picks of wide steps had no slot, {} of them \
             computed on the CPU ({} pairs); a layer's rows differ from the run without misses \
             by at most {:.2e} (experts), {:.2e} (core), {:.2e} (residual), relative RMS",
            devices.len(),
            wide_stats.missing,
            wide_stats.picked,
            wide_stats.cpu_experts,
            wide_stats.cpu_pairs,
            worst[0],
            worst[1],
            worst[2],
        );
        assert!(wide_stats.missing > 100, "the wide steps were to miss experts");
        if serve == Serve::Cpu {
            assert!(wide_stats.cpu_experts > 0, "the CPU took the misses it could");
            assert!(stats.cpu_experts >= wide_stats.cpu_experts);
        }
        assert_eq!(worst[1], 0.0, "{serve:?}: the core reads the same residual");
        assert!(worst[0] < 4e-2, "{serve:?}: experts {:.2e}", worst[0]);
        assert!(worst[2] < 2e-2, "{serve:?}: residual {:.2e}", worst[2]);
    }
}

/// Feeds every layer the reference's input and compares what its experts give row by row.
///
/// A row is compared when its experts read what the reference's read and are the
/// reference's: `worst` is the largest relative RMS of such a row's expert output and
/// residual. A row is counted and not compared when the discrete choices before its experts
/// went the other way at a near tie, which the rounding of the products before them can
/// cause (8-bit activations in steps of up to eight rows, f16 operands of the float
/// matrices in wider ones): its router picked another expert, or its attention selected
/// another block, which shows as a core far from the reference's ([`OTHER_BLOCK`]).
struct ExpertRows<'a> {
    expected: &'a [Expected],
    n: usize,
    /// Expert output and residual of the rows that are compared.
    worst: [f64; 2],
    rows: usize,
    compared: usize,
}

/// A row's core is this far from the reference's (relative RMS) only when its attention
/// read other blocks: with the reference's blocks it is within 3e-2
/// (`every_layer_matches_the_reference`), with another block it is another vector.
const OTHER_BLOCK: f64 = 0.1;

impl Probe for ExpertRows<'_> {
    fn before(&mut self, layer: usize) -> Option<Vec<f32>> {
        Some(self.expected[layer].input.clone())
    }

    fn after(&mut self, layer: usize, out: LayerOut<'_>) {
        let want = &self.expected[layer];
        let n = self.n;
        let wide = out.residual.len() / out.picks.len();
        for (row, (got, reference)) in out.picks.iter().zip(&want.picks).enumerate() {
            self.rows += 1;
            let (mut got, mut reference) = (got.clone(), reference.clone());
            got.sort_unstable();
            reference.sort_unstable();
            let at = row * n..(row + 1) * n;
            let streams = row * wide..(row + 1) * wide;
            let errors = [
                rel_rms(&out.core[at.clone()], &want.core[at.clone()]),
                rel_rms(&out.ffn[at.clone()], &want.ffn[at]),
                rel_rms(&out.residual[streams.clone()], &want.output[streams]),
            ];
            // A maximum drops what is not a number.
            assert!(
                errors.iter().all(|e| e.is_finite()),
                "layer {layer}, row {row}: an output that is not a number"
            );
            if got == reference && errors[0] < OTHER_BLOCK {
                self.compared += 1;
                for (worst, error) in self.worst.iter_mut().zip(&errors[1..]) {
                    *worst = worst.max(*error);
                }
            }
        }
    }
}

/// Models with experts of the pairs `part * 7 ..` of [`type_pairs`], in steps of up to eight
/// rows (the decode kernels) and of 64 and 20 rows (the many-row kernels), against the CPU
/// reference of the model.
///
/// First every layer by itself, fed the reference's input, row by row ([`ExpertRows`]):
/// what the experts of a type give is held to the reference there. Then a free run, whose
/// logits are what those differences grow to through the layers of a made-up model (its
/// 8-bit activation codes and its discrete choices amplify them, see
/// `experts_computed_on_the_cpu_change_nothing`): they must be numbers, and the most likely
/// token must be one the reference's logits put near the top.
fn expert_types_match_the_reference(part: usize) {
    let Some(devices) = devices() else { return };
    for (gate_ty, down_ty) in type_pairs().into_iter().skip(7 * part).take(7) {
        let synthetic = model_of(41, down_ty);
        let name = format!("types-{gate_ty}-{down_ty}");
        let gguf = open(&name, &synthetic.bytes_with_experts(gate_ty, down_ty));
        let c = Config::from_gguf(&gguf).unwrap();
        let w = Weights::load(&gguf, &c).unwrap();
        let reference = Reference::new(&c, &w);
        let steps = [8usize, 3, 64, 20, 1];
        let prompt = synthetic.shape.tokens(steps.iter().sum(), 9);
        let open_session = || {
            let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
            opts.model.part.max_rows = 64;
            let mut session =
                Session::open(&gguf, &c, &w, devices.clone(), opts).expect("the model opens");
            let n_expert = c.experts.count as u32;
            session.fill(&mut |_| (0..n_expert).collect()).expect("the tiers fill");
            session
        };

        // Layer by layer: [up to eight rows, more][expert output, residual] of the rows
        // that are compared.
        let mut session = open_session();
        let mut state = SequenceState::new(&c);
        let mut worst = [[0.0f64; 2]; 2];
        let (mut rows, mut compared, mut at) = (0, 0, 0);
        for count in steps {
            let tokens = &prompt[at..at + count];
            at += count;
            let wide = usize::from(count > 8);
            let layers = expected(&reference, &c, &w, &state, tokens);
            let mut probe = ExpertRows {
                expected: &layers,
                n: c.n_embd,
                worst: worst[wide],
                rows: 0,
                compared: 0,
            };
            session
                .step_probed(
                    &[Rows { slot: 0, tokens, direct: true, wanted: Wanted::None }],
                    &mut probe,
                )
                .unwrap();
            worst[wide] = probe.worst;
            rows += probe.rows;
            compared += probe.compared;
            let window = reference.window(&state, tokens, LogitRows::None).unwrap();
            reference.commit(&mut state, &window, count).unwrap();
        }
        drop(session);

        // A free run.
        let mut session = open_session();
        let mut state = SequenceState::new(&c);
        let (mut at, mut logits) = (0, [0.0f64; 2]);
        for count in steps {
            let tokens = &prompt[at..at + count];
            at += count;
            let picks = session
                .step(&[Rows { slot: 0, tokens, direct: true, wanted: Wanted::Last }])
                .unwrap()
                .to_vec();
            let window = reference.window(&state, tokens, LogitRows::All).unwrap();
            let want = window.logits_of(count - 1).unwrap();
            let got = session.logits(0).unwrap();
            let error = rel_rms(got, want);
            // A maximum drops what is not a number.
            assert!(
                error.is_finite() && got.iter().all(|v| v.is_finite()),
                "{gate_ty} and {down_ty}: the logits after {at} tokens are not numbers"
            );
            assert!(
                is_near_best(picks[0], want, error),
                "{gate_ty} and {down_ty}: the most likely token after {at} tokens ({error:.2e})"
            );
            let wide = usize::from(count > 8);
            logits[wide] = logits[wide].max(error);
            reference.commit(&mut state, &window, count).unwrap();
        }
        eprintln!(
            "  ERR {gate_ty} and {down_ty}, {} device(s): the experts of a row differ from the \
             reference's by at most {:.2e} (up to eight rows) and {:.2e} (64 and 20 rows), the \
             residual by {:.2e} and {:.2e}, in the {compared} of {rows} rows of layers that made \
             the reference's choices; the logits of a free run by {:.2e} and {:.2e} (relative \
             RMS)",
            devices.len(),
            worst[0][0],
            worst[1][0],
            worst[0][1],
            worst[1][1],
            logits[0],
            logits[1],
        );
        for (wide, (experts, residual)) in [EXPERT_ROW_8BIT, EXPERT_ROW_F16].iter().enumerate() {
            let what = format!("{gate_ty} and {down_ty}, {}", ["up to eight rows", "wide"][wide]);
            assert!(worst[wide][0] < *experts, "{what}: experts {:.2e}", worst[wide][0]);
            assert!(worst[wide][1] < *residual, "{what}: residual {:.2e}", worst[wide][1]);
        }
        assert!(compared * 100 >= rows * 97, "{compared} of {rows} rows could be compared");
    }
}

/// The matrices the device multiplies unquantized (the stream mixers' down, up and inject,
/// the n-gram block's key and value) in every weight type, each layer of steps of up to eight
/// rows (the fused read and the exact decode kernel) and of a wide step (the type's tile
/// kernel) fed the reference's input, where the reference decodes the same weights. The rank
/// of the mixers is 256 and every row width a multiple of 256, so that every type can hold
/// them; the n-gram table takes the type too where it can hold rows of 32 values (the 32-value
/// types the writer encodes), else IQ4_NL. (The kernels themselves give the bits of the
/// decoded weights in F32: `crates/cuda/tests/{mmvq,q4}.rs`.)
///
/// Each type is a model of its own, and the bounds are those of a model's numbers: on
/// 2026-11-17 (one card, the 28 types) the recurrent cores differed by 3.6e-4 to 2.3e-3, the
/// residuals by at most 2.4e-2, and the experts by up to 0.21 (IQ1_S: a row whose picks flip
/// at a near tie, as the experts take 8-bit inputs); an attention core of a wide step
/// differed by 0.22 in one model (Q1_0), where a row past the dense limit selected another
/// block than the reference at a near tie of the indexer's scores (f16 keys on the device),
/// with its residual within the bound. The experts are held to the bound of a wide step of
/// quantized dense matrices (0.3). All types are run and printed before a bound is checked.
fn float_path_types_match_the_reference(part: usize) {
    let Some(devices) = devices() else { return };
    let types: Vec<GgmlType> = ExpertShape::types().collect();
    let table_types = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
    ];
    let mut failed = Vec::new();
    for ty in types.into_iter().skip(7 * part).take(7) {
        let mut synthetic = model(23);
        synthetic.shape.mixer_rank = 256;
        synthetic.lazy_type = Some(if table_types.contains(&ty) { ty } else { GgmlType::IQ4_NL });
        let gguf = open(
            &format!("float-path-{}", ty.name()),
            &synthetic.bytes_with_types(Some((GgmlType::IQ2_S, GgmlType::Q2_0)), ty),
        );
        let c = Config::from_gguf(&gguf).unwrap();
        let w = Weights::load(&gguf, &c).unwrap();
        assert_eq!(w.layers[0].attn_mixer.down.ty(), ty);
        let reference = Reference::new(&c, &w);
        let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
        opts.model.part.max_rows = 64;
        let mut session =
            Session::open(&gguf, &c, &w, devices.clone(), opts).expect("the model opens");
        let n_expert = c.experts.count as u32;
        session.fill(&mut |_| (0..n_expert).collect()).expect("the tiers fill");
        let steps = [8usize, 5, 1, 17, 3];
        let all = synthetic.shape.tokens(steps.iter().sum(), 3);
        let mut state = SequenceState::new(&c);
        // Per kind of step (up to eight rows, wide), per kind of layer.
        let mut worst = [[[0.0f64; 3]; 2]; 2];
        let (mut rows, mut same_picks, mut at) = (0, 0, 0);
        for &count in &steps {
            let tokens = &all[at..at + count];
            let wide = usize::from(count > 8);
            let layers = expected(&reference, &c, &w, &state, tokens);
            let mut probe = Compare {
                expected: &layers,
                attention: &c.attention_layers,
                worst: worst[wide],
                rows: 0,
                same_picks: 0,
            };
            session
                .step_probed(
                    &[Rows { slot: 0, tokens, direct: true, wanted: Wanted::None }],
                    &mut probe,
                )
                .unwrap();
            worst[wide] = probe.worst;
            rows += probe.rows;
            same_picks += probe.same_picks;
            let window = reference.window(&state, tokens, LogitRows::None).unwrap();
            reference.commit(&mut state, &window, count).unwrap();
            at += count;
        }
        for (wide, worst) in worst.iter().enumerate() {
            let what = format!("{ty}, {}", ["up to eight rows", "wide"][wide]);
            eprintln!(
                "  ERR {what}, {} device(s): recurrent layers: core {:.2e}, experts {:.2e}, \
                 residual {:.2e}; attention layers: core {:.2e}, experts {:.2e}, residual {:.2e}",
                devices.len(),
                worst[0][0],
                worst[0][1],
                worst[0][2],
                worst[1][0],
                worst[1][1],
                worst[1][2],
            );
            let mut check = |ok: bool, text: String| {
                if !ok {
                    failed.push(format!("{what}: {text}"));
                }
            };
            check(worst[0][0] < 5e-3, format!("recurrent core {:.2e}", worst[0][0]));
            check(worst[1][0] < 0.3, format!("attention core {:.2e}", worst[1][0]));
            for kind in worst {
                check(kind[1] < 0.3, format!("experts {:.2e}", kind[1]));
                check(kind[2] < 4e-2, format!("residual {:.2e}", kind[2]));
            }
        }
        if same_picks * 100 < rows * 95 {
            failed.push(format!("{ty}: {same_picks} of {rows} routings"));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn float_path_matrices_of_every_weight_type_match_the_reference_1() {
    float_path_types_match_the_reference(0);
}

#[test]
fn float_path_matrices_of_every_weight_type_match_the_reference_2() {
    float_path_types_match_the_reference(1);
}

#[test]
fn float_path_matrices_of_every_weight_type_match_the_reference_3() {
    float_path_types_match_the_reference(2);
}

#[test]
fn float_path_matrices_of_every_weight_type_match_the_reference_4() {
    float_path_types_match_the_reference(3);
}

/// The bounds of one row's expert output and of its residual against the reference's
/// (relative RMS). In steps of up to eight rows the decode kernels round the activations and
/// the hidden values to 8 bits: 1.4e-2 to 2.8e-2 and 3.4e-3 to 6.0e-3 over the 28 pairs of
/// types, on one card and on two (2026-11-12). In wider steps the many-row kernels round
/// both to f16 and the dense products before them read f16 operands: 5.9e-3 to 1.3e-2 and
/// 2.3e-3 to 3.6e-3. 764 to 768 of a model's 768 rows of layers were compared.
const EXPERT_ROW_8BIT: (f64, f64) = (8e-2, 2e-2);
const EXPERT_ROW_F16: (f64, f64) = (4e-2, 1e-2);

#[test]
fn experts_of_every_weight_type_match_the_reference_1() {
    expert_types_match_the_reference(0);
}

#[test]
fn experts_of_every_weight_type_match_the_reference_2() {
    expert_types_match_the_reference(1);
}

#[test]
fn experts_of_every_weight_type_match_the_reference_3() {
    expert_types_match_the_reference(2);
}

#[test]
fn experts_of_every_weight_type_match_the_reference_4() {
    expert_types_match_the_reference(3);
}

/// An expert of any weight type that is in no slot is computed on the CPU with the result
/// the device gives, within the bound of the types of the real file
/// (`experts_computed_on_the_cpu_change_nothing`).
fn expert_types_on_the_cpu_change_nothing(part: usize) {
    let Some(devices) = devices() else { return };
    for (gate_ty, down_ty) in type_pairs().into_iter().skip(7 * part).take(7) {
        let [experts, core, residual] =
            layers_with_misses_of(&devices, Serve::Cpu, gate_ty, down_ty);
        assert_eq!(core, 0.0, "{gate_ty} and {down_ty}");
        assert!(experts < 1e-2, "{gate_ty} and {down_ty}: experts {experts:.2e}");
        assert!(residual < 1e-2, "{gate_ty} and {down_ty}: residual {residual:.2e}");
    }
}

#[test]
fn experts_of_every_weight_type_on_the_cpu_change_nothing_1() {
    expert_types_on_the_cpu_change_nothing(0);
}

#[test]
fn experts_of_every_weight_type_on_the_cpu_change_nothing_2() {
    expert_types_on_the_cpu_change_nothing(1);
}

#[test]
fn experts_of_every_weight_type_on_the_cpu_change_nothing_3() {
    expert_types_on_the_cpu_change_nothing(2);
}

#[test]
fn experts_of_every_weight_type_on_the_cpu_change_nothing_4() {
    expert_types_on_the_cpu_change_nothing(3);
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
    opts.model.part.profile = Some(profile.clone());
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
    opts.model.slot_bytes = Some(slots * shape.slot_bytes().unwrap());
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
        let mut opts = options(devices.len(), c.n_layer, CacheFormat::Q8);
        opts.model.slot_bytes = Some(slots * shape.slot_bytes().unwrap());
        opts.model.serve = serve;
        opts.model.stall = std::time::Duration::from_secs(20);
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

/// The routers' pruning (`device/prune.rs`): with every expert in a tier it
/// touches nothing, and the logits are those of a session without it, bit for
/// bit; with most experts only in the file, each rule leaves out or stands in
/// for what the file alone holds, so that no step waits for a read, and every
/// logit stays a number.
#[test]
fn pruning_touches_only_what_the_file_alone_holds() {
    let Some(devices) = devices() else { return };
    let mut synthetic = model(53);
    synthetic.shape.n_expert = if tensor().is_some() { 384 } else { 192 };
    let gguf = open("prune", &synthetic.bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0));
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let text = synthetic.shape.tokens(48, 17);
    let run = |session: &mut Session<'_>| -> Vec<Vec<f32>> {
        let mut logits = Vec::new();
        for tokens in text[..16].chunks(8) {
            session.step(&[Rows { slot: 0, tokens, direct: true, wanted: Wanted::All }]).unwrap();
            logits.extend((0..tokens.len()).map(|row| session.logits(row).unwrap().to_vec()));
        }
        for token in text[16..].chunks(1) {
            session
                .step(&[Rows { slot: 0, tokens: token, direct: true, wanted: Wanted::Last }])
                .unwrap();
            logits.push(session.logits(0).unwrap().to_vec());
        }
        logits
    };
    let n_expert = c.experts.count;
    // Each expert's most similar ones: its next numbers, nearer first.
    let key = ProfileKey::new(b"made-up model", c.n_layer as u32, n_expert as u32);
    let mut near = ExpertNeighbours::new(key, 4);
    for layer in 0..c.n_layer {
        for expert in 0..n_expert {
            for (i, entry) in near.of_mut(layer, expert).iter_mut().enumerate() {
                let other = (expert + 1 + 7 * i) % n_expert;
                *entry = Near { expert: other as u16, closeness: 0.9 - 0.1 * i as f32, scale: 1.0 };
            }
        }
    }
    let near = Arc::new(near);
    let setup = |rule: PruneRule, weights: PruneWeights| PruneSetup {
        prune: Prune { weights, ..Prune::new(rule, 0) },
        near: Some(near.clone()),
    };
    let base = options(devices.len(), c.n_layer, CacheFormat::Q8);

    // Every expert in VRAM: nothing to prune.
    let want = run(&mut session(&gguf, &c, &w, devices.clone(), CacheFormat::Q8));
    let mut opts = base.clone();
    opts.model.part.prune = Some(setup(PruneRule::Skip, PruneWeights::Keep));
    let mut pruned = Session::open(&gguf, &c, &w, devices.clone(), opts).unwrap();
    pruned.fill(&mut |_| (0..n_expert as u32).collect()).unwrap();
    pruned.take_prune_stats();
    let got = run(&mut pruned);
    assert_eq!(got, want, "pruning touched a pick whose expert has a slot");
    let stats = pruned.take_prune_stats();
    assert!(stats.picks > 0);
    assert_eq!(stats.touched + stats.stood_in, 0);
    drop(pruned);

    // A slot per pick a step can need in VRAM, as few in RAM: most experts
    // are only in the file.
    let shape = ExpertShape {
        n_embd: c.n_embd as u32,
        n_ff: c.experts.ff as u32,
        n_experts: n_expert as u32,
        top_k: c.experts.used as u32,
        gate_ty: GgmlType::IQ2_S,
        down_ty: GgmlType::Q2_0,
    };
    let per_layer = c.experts.used * 8;
    let slots = (per_layer * layers_per_device(c.n_layer, devices.len())) as u64;
    for (rule, weights) in [
        (PruneRule::Skip, PruneWeights::Keep),
        (PruneRule::Skip, PruneWeights::Renormalise),
        (PruneRule::Router, PruneWeights::Renormalise),
        (PruneRule::Similar, PruneWeights::Keep),
        (PruneRule::Similar, PruneWeights::Scaled),
    ] {
        let mut opts = base.clone();
        opts.model.part.prune = Some(setup(rule, weights));
        opts.model.slot_bytes = Some(slots * shape.slot_bytes().unwrap());
        opts.model.serve = Serve::Stream;
        opts.ram_tier = Some(slots * shape.slot_bytes().unwrap() * 2);
        let mut session = Session::open(&gguf, &c, &w, devices.clone(), opts).unwrap();
        session.fill(&mut |_| (0..n_expert as u32).collect()).unwrap();
        session.take_store_stats();
        session.take_prune_stats();
        let got = run(&mut session);
        let store = session.take_store_stats();
        let pruned = session.take_prune_stats();
        eprintln!(
            "  {} device(s), {rule:?} with {weights:?} weights: {} picks, {} left out, {} stood \
             in for; {} in a slot, {} from RAM, {} from the file",
            devices.len(),
            pruned.picks,
            pruned.left_out(),
            pruned.stood_in,
            store.hits,
            store.miss_from_ram,
            store.miss_from_disk,
        );
        assert!(got.iter().flatten().all(|v| v.is_finite()), "{rule:?}: a logit that is no number");
        assert!(pruned.touched > 0, "{rule:?} touched nothing");
        match rule {
            // Every pick only the file holds is left out or stood in for.
            PruneRule::Skip | PruneRule::Router => assert_eq!(store.miss_from_disk, 0),
            // A pick none of whose similar experts is in a tier stays.
            PruneRule::Similar => assert!(pruned.stood_in > 0),
        }
        if rule != PruneRule::Skip {
            assert_eq!(pruned.left_out(), 0, "{rule:?} left a pick out");
        }
    }
}
