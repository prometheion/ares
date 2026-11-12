//! What the expert path of a mixture-of-experts model allows on its own:
//! the routers and the routed experts of every layer of a decode step, with
//! the experts in tiers (VRAM slots, pinned RAM, the file on the SSD) and
//! brought in by the expert store while the step runs.
//!
//! The model is Qwen3.8-Flash-Next: its real routers and expert tensors, read
//! from the real file. What is not real is everything around the experts
//! (attention, recurrent layers, mixers), so the routing cannot come from the
//! model. It comes from a routing trace instead: a recorded one, or a
//! synthetic one whose skew and drift are parameters. The picks of the trace
//! are given to the device as logits, so that the device's own router kernel
//! finds them and the host learns them only from the route record, exactly
//! as in a real step. The real router matrix is multiplied as well, for its
//! cost.
//!
//! A step is enqueued whole on every device (the layers are split over the
//! devices in order), the route records come back on a side stream, and the
//! kernels of an expert that has no slot wait on the device until the store
//! has uploaded it. The numbers:
//!
//! - steps and rows per second, first from empty tiers, then warm;
//! - bytes read from the file per step, and the share of picks found in a slot;
//! - how long a missing expert took (median and 99th percentile), by where
//!   it came from.
//!
//! Predictors are emulated with a given recall, since there is no hidden
//! state to apply a router to: `lookahead=<d>` names the picks of layer
//! `l + d` when layer `l` has been routed, `table=<layers>` names the picks
//! of the first layers before a step starts. Their kernels' cost is real.
//!
//! ```sh
//! cargo run --release -p ares-runtime --example expert_pipe_bench -- \
//!     model=<model-00001-of-00002.gguf> devices=0,1 vram-gib=9,9 ram-gib=8 rows=1
//! ```
//!
//! `sim=1` runs no device and no I/O: it prints the share of picks a cache of
//! a given size serves on the trace, which is how the synthetic trace is
//! compared with hit rates reported for the real model.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, ExpertKernels, ExpertShape, Kernels, PinnedBuffer,
    RouteGroup, RouteRecord, Stream, Wait,
};
use ares_gguf::Gguf;
use ares_io::{Config, ExpertMap, Reader};
use ares_runtime::expert_store::cpu::act_bytes;
use ares_runtime::expert_store::cpu::{
    CpuExpert, CpuPool, Decode, Features, LayerJob, Placement, Topology, quantize_activations,
};
use ares_runtime::expert_store::{
    CudaSlots, ExpertStore, HostRegion, LayerSpec, Miss, Serve, StoreConfig, StorePlan, StoreStats,
};
use ares_weights::{ExpertHeat, SlotSet, Start, TableBuilder, TableConfig, Want};

type AnyError = Box<dyn std::error::Error>;

/// A small deterministic generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.unit() + 1e-12, self.unit());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// A distribution over `0..n` where `i` has weight `(i + 1)^-s`.
struct Zipf(Vec<f64>);

impl Zipf {
    fn new(n: usize, s: f64) -> Self {
        let mut total = 0.0;
        let mut cdf: Vec<f64> = (0..n)
            .map(|i| {
                total += ((i + 1) as f64).powf(-s);
                total
            })
            .collect();
        cdf.iter_mut().for_each(|c| *c /= total);
        Self(cdf)
    }

    fn at(&self, u: f64) -> usize {
        self.0.partition_point(|c| *c <= u).min(self.0.len() - 1)
    }
}

/// The routing of a text: for every token and layer the experts the router
/// picked, the largest weight first.
struct Trace {
    layers: usize,
    k: usize,
    tokens: Vec<u32>,
    picks: Vec<u16>,
    /// Where the routing came from, for the report.
    origin: String,
}

impl Trace {
    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn picks(&self, token: usize, layer: usize) -> &[u16] {
        &self.picks[(token * self.layers + layer) * self.k..][..self.k]
    }

    /// A synthetic text over topics. A topic lasts 300 to 900 tokens and has
    /// a group of `group` experts in every layer; `share` of a token's picks
    /// come from the group (ranked inside it), the others from a popularity
    /// over all experts that is skewed by `skew` (a Zipf exponent). In the
    /// first `early` layers the picks follow the token instead: `by_token`
    /// of them are fixed by the token's id. The same structure as the
    /// workload of `ares-weights`' premise tests, with its numbers open.
    #[allow(clippy::too_many_arguments)]
    fn synthetic(
        layers: usize,
        experts: usize,
        k: usize,
        n: usize,
        early: usize,
        share: f64,
        group: usize,
        skew: f64,
        by_token: f64,
        seed: u64,
    ) -> Self {
        const TOPICS: usize = 6;
        const VOCAB: usize = 4000;
        let mut rng = Rng::new(seed);
        let popularity: Vec<(Zipf, Vec<u16>)> = (0..layers)
            .map(|l| {
                let s = if l < early { 0.3 } else { skew };
                let mut order: Vec<u16> = (0..experts as u16).collect();
                for i in (1..order.len()).rev() {
                    order.swap(i, rng.below(i + 1));
                }
                (Zipf::new(experts, s), order)
            })
            .collect();
        let pick = |l: usize, u: f64| popularity[l].1[popularity[l].0.at(u)];
        let groups: Vec<Vec<Vec<u16>>> = (0..layers)
            .map(|l| {
                (0..TOPICS)
                    .map(|_| {
                        let mut members: Vec<u16> = Vec::new();
                        while members.len() < group {
                            let e = pick(l, rng.unit());
                            if !members.contains(&e) {
                                members.push(e);
                            }
                        }
                        members
                    })
                    .collect()
            })
            .collect();
        let in_group = Zipf::new(group, 0.8);
        let (function_words, own_words) = (Zipf::new(60, 1.0), Zipf::new(250, 1.0));
        let words: Vec<Vec<u32>> = (0..TOPICS)
            .map(|_| (0..250).map(|_| (60 + rng.below(VOCAB - 60)) as u32).collect())
            .collect();
        let hash = |parts: [u64; 3]| {
            let mut x = 0x243f_6a88_85a3_08d3u64;
            for part in parts {
                x ^= part.wrapping_add(0x9e37_79b9_7f4a_7c15);
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                x ^= x >> 31;
            }
            (x >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut tokens = Vec::with_capacity(n);
        let mut picks = Vec::with_capacity(n * layers * k);
        let (mut topic, mut until) = (0usize, 300 + rng.below(600));
        for pass in 0..n {
            if pass >= until {
                topic = (topic + 1 + rng.below(TOPICS - 1)) % TOPICS;
                until = pass + 300 + rng.below(600);
            }
            let token = if rng.unit() < 0.4 {
                function_words.at(rng.unit()) as u32
            } else {
                words[topic][own_words.at(rng.unit())]
            };
            for (l, layer_groups) in groups.iter().enumerate() {
                let mut row: Vec<u16> = Vec::with_capacity(k);
                for i in 0..k {
                    let mut expert = if l < early {
                        if rng.unit() < by_token {
                            pick(l, hash([l as u64, u64::from(token), i as u64]))
                        } else {
                            pick(l, rng.unit())
                        }
                    } else if rng.unit() < share {
                        layer_groups[topic][in_group.at(rng.unit())]
                    } else {
                        pick(l, rng.unit())
                    };
                    while row.contains(&expert) {
                        expert = pick(l, rng.unit());
                    }
                    row.push(expert);
                }
                picks.extend_from_slice(&row);
            }
            tokens.push(token);
        }
        let origin = format!(
            "synthetic (share {share}, group {group}, skew {skew}, early {early}, by-token {by_token}, seed {seed})"
        );
        Self { layers, k, tokens, picks, origin }
    }

    /// Recorded traces (`*.q4route` of `/var/tmp/ares-qwen4exp-trace/`), one
    /// after the other. A file is a header of 64 bytes ("Q4ROUTE1", then u32:
    /// version, layers, experts, k, tokens, ...), the token ids (u32), and
    /// for every token and layer `k` experts (u16), the largest weight
    /// first, followed by their `k` weights (f32). (Read from the files; the
    /// directory had no README when this was written.)
    fn load(paths: &str, layers: usize, k: usize) -> Result<Self, AnyError> {
        let mut tokens = Vec::new();
        let mut picks = Vec::new();
        for path in paths.split(',') {
            let bytes = std::fs::read(path)?;
            let word = |at: usize| {
                bytes
                    .get(at..at + 4)
                    .map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
            };
            let n = word(24);
            let per_layer = k * 6;
            if bytes.get(..8) != Some(b"Q4ROUTE1")
                || (word(12), word(20)) != (layers, k)
                || bytes.len() != 64 + n * (4 + layers * per_layer)
            {
                return Err(format!(
                    "{path}: not a routing trace of {layers} layers and {k} picks"
                )
                .into());
            }
            tokens.extend((0..n).map(|t| word(64 + 4 * t) as u32));
            for record in bytes[64 + 4 * n..].chunks_exact(per_layer) {
                picks.extend(
                    record[..2 * k].as_chunks::<2>().0.iter().map(|id| u16::from_le_bytes(*id)),
                );
            }
        }
        Ok(Self { layers, k, tokens, picks, origin: format!("recorded ({paths})") })
    }

    /// What the trace says about locality: the share of a token's picks that
    /// the token before it picked in the same layer, and the experts a layer
    /// uses over the whole trace (fewest, mean, most).
    fn locality(&self, experts: usize) -> (f64, usize, f64, usize) {
        let (mut same, mut all) = (0u64, 0u64);
        let mut distinct = Vec::with_capacity(self.layers);
        for layer in 0..self.layers {
            let mut used = vec![false; experts];
            for token in 0..self.len() {
                let picks = self.picks(token, layer);
                picks.iter().for_each(|e| used[*e as usize] = true);
                if token > 0 {
                    let before = self.picks(token - 1, layer);
                    same += picks.iter().filter(|e| before.contains(e)).count() as u64;
                    all += picks.len() as u64;
                }
            }
            distinct.push(used.iter().filter(|u| **u).count());
        }
        let mean = distinct.iter().sum::<usize>() as f64 / self.layers as f64;
        (
            same as f64 / all.max(1) as f64,
            distinct.iter().copied().min().unwrap_or(0),
            mean,
            distinct.iter().copied().max().unwrap_or(0),
        )
    }
}

/// The share of picks a cache of `slots` experts per layer serves on the
/// trace, fed by misses as the store feeds a tier, for steps of `rows` tokens.
fn simulate(trace: &Trace, experts: usize, slots: usize, rows: usize, skip: usize) -> f64 {
    let mut heat = ExpertHeat::new(trace.layers, experts, 0.9);
    let mut tiers: Vec<SlotSet> = (0..trace.layers).map(|_| SlotSet::new(experts, slots)).collect();
    let (mut picks, mut hits) = (0u64, 0u64);
    for (step, first) in (0..trace.len() - rows + 1).step_by(rows).enumerate() {
        for (layer, tier) in tiers.iter_mut().enumerate() {
            let mut routed: Vec<u32> = (first..first + rows)
                .flat_map(|t| trace.picks(t, layer).iter().map(|e| u32::from(*e)))
                .collect();
            routed.sort_unstable();
            routed.dedup();
            heat.record(layer, &routed);
            tier.routed(&routed, step as u64);
            for expert in routed {
                let start = tier.start_read(expert, Want::Demand, step as u64, heat.layer(layer));
                if step >= skip {
                    picks += 1;
                    hits += u64::from(start == Start::Present);
                }
                if let Start::Slot(swap) = start {
                    tier.commit(&swap);
                }
            }
        }
        if step % 32 == 31 {
            heat.decay();
        }
    }
    hits as f64 / picks.max(1) as f64
}

/// What a token table says about the first `layers` layers of the trace: a
/// table built from the first `train` tokens, asked for every later token.
/// Returns the share of those tokens the table knows, and the precision and
/// recall of what it names for them (recall over all later tokens).
fn table_quality(trace: &Trace, experts: usize, layers: usize, train: usize) -> (f64, f64, f64) {
    let mut builder = TableBuilder::new(TableConfig::new(layers, trace.k), experts);
    let routed = |token: usize| -> Vec<Vec<u32>> {
        (0..layers).map(|l| trace.picks(token, l).iter().map(|e| u32::from(*e)).collect()).collect()
    };
    for token in 0..train {
        let previous = token.checked_sub(1).map(|t| trace.tokens[t]);
        builder.record(previous, trace.tokens[token], &routed(token));
    }
    let table = builder.build();
    let (mut known, mut named, mut right, mut wanted) = (0u64, 0u64, 0u64, 0u64);
    for token in train..trace.len() {
        wanted += (layers * trace.k) as u64;
        let Some(entry) = table.lookup(Some(trace.tokens[token - 1]), trace.tokens[token]) else {
            continue;
        };
        known += 1;
        for (layer, picks) in routed(token).iter().enumerate() {
            for (expert, _) in entry.layer(layer) {
                named += 1;
                right += u64::from(picks.contains(&expert));
            }
        }
    }
    (
        known as f64 / (trace.len() - train).max(1) as f64,
        right as f64 / named.max(1) as f64,
        right as f64 / wanted.max(1) as f64,
    )
}

/// An emulated predictor: names each pick with the probability its rank has,
/// and a wrong expert in its place otherwise.
fn predict(picks: &[u16], recall_by_rank: &[f64], experts: usize, rng: &mut Rng) -> Vec<u32> {
    picks
        .iter()
        .enumerate()
        .map(|(rank, pick)| {
            if rng.unit() < recall_by_rank[rank.min(recall_by_rank.len() - 1)] {
                u32::from(*pick)
            } else {
                loop {
                    let wrong = rng.below(experts) as u16;
                    if !picks.contains(&wrong) {
                        break u32::from(wrong);
                    }
                }
            }
        })
        .collect()
}

/// What an emulated predictor names for `layer` of the step whose rows start
/// at `token`: the experts of all rows, the surest rank first, and how many
/// of them are of the first `read_ranks` ranks (those may be read from the
/// file).
fn named_by(
    trace: &Trace,
    token: usize,
    rows: usize,
    layer: usize,
    recall_by_rank: &[f64],
    (name_ranks, read_ranks): (usize, usize),
    rng: &mut Rng,
) -> (Vec<u32>, usize) {
    let per_row: Vec<Vec<u32>> = (0..rows)
        .map(|row| predict(trace.picks(token + row, layer), recall_by_rank, 512, rng))
        .collect();
    let mut named: Vec<u32> = Vec::with_capacity(rows * trace.k);
    let mut sure = 0;
    for rank in 0..name_ranks.min(trace.k) {
        for row in &per_row {
            if !named.contains(&row[rank]) {
                named.push(row[rank]);
            }
        }
        if rank + 1 == read_ranks {
            sure = named.len();
        }
    }
    if read_ranks >= name_ranks.min(trace.k) {
        sure = named.len();
    }
    (named, sure)
}

/// The recall per rank of a router applied one layer early, as Strata's
/// notes report it for the first six ranks, continued downwards; further
/// layers ahead scale by the recalls Run My AI reports (75.7, 67.2, 61.1 %).
fn lookahead_recall(distance: usize) -> Vec<f64> {
    let scale = [1.0, 0.888, 0.807, 0.74, 0.68][distance.clamp(1, 5) - 1];
    [0.97, 0.91, 0.81, 0.70, 0.57, 0.41, 0.35, 0.30, 0.25, 0.20].iter().map(|r| r * scale).collect()
}

struct Options {
    model: PathBuf,
    trace: Option<String>,
    devices: Vec<u32>,
    vram_gib: Vec<f64>,
    ram_gib: f64,
    rows: u32,
    cold: usize,
    warm: usize,
    steps: usize,
    lookahead: usize,
    table: usize,
    table_recall: f64,
    dense_mb: u64,
    reserve: usize,
    lanes: usize,
    profile: usize,
    read_ranks: usize,
    name_ranks: usize,
    sweep: Vec<(usize, usize)>,
    prompt: usize,
    wave_room: f64,
    share: f64,
    group: usize,
    skew: f64,
    seed: u64,
    sim: bool,
    cpu_bench: bool,
    /// How a miss is served: streamed, computed on the CPU, or by the cost model.
    serve: Serve,
    /// The logical CPUs of the workers; none leaves it to the store's placement.
    cpus: Option<Vec<usize>>,
    /// How long an idle worker spins before it sleeps, in milliseconds.
    spin_ms: u64,
    /// After every step, fold the rows' outputs into a checksum (outside the
    /// timing): two runs over the same tokens that serve misses differently
    /// must agree in it to rounding.
    verify: bool,
    /// With `verify`: write every step's output of every device to this
    /// file, or compare every step's with the file another run wrote. A
    /// checksum over a run says that the runs agree in the sum; this says
    /// which step does not.
    verify_save: Option<String>,
    verify_against: Option<String>,
    /// In the measured phase, serve misses streamed, on the CPU and by the
    /// cost model in turns of this many steps, on one store: the three then
    /// see the same drive and the same machine, which runs after each other
    /// do not when both are shared. 0 serves one way.
    alternate: usize,
    /// With `serve=stream`: start the workers all the same and leave them
    /// without work, to see what their spinning takes from the step's own
    /// threads.
    idle_pool: bool,
}

impl Options {
    fn parse() -> Result<Self, AnyError> {
        let mut options = Self {
            model: PathBuf::new(),
            trace: None,
            devices: vec![0],
            vram_gib: vec![8.0],
            ram_gib: 4.0,
            rows: 1,
            cold: 16,
            warm: 200,
            steps: 300,
            lookahead: 0,
            table: 0,
            table_recall: 0.6,
            dense_mb: 0,
            reserve: 8,
            lanes: 8,
            profile: 0,
            read_ranks: 2,
            name_ranks: 10,
            sweep: Vec::new(),
            prompt: 0,
            wave_room: 0.5,
            share: 0.55,
            group: 24,
            skew: 0.9,
            seed: 1,
            sim: false,
            cpu_bench: false,
            serve: Serve::Measured,
            cpus: None,
            spin_ms: 20,
            verify: false,
            verify_save: None,
            verify_against: None,
            alternate: 0,
            idle_pool: false,
        };
        for arg in std::env::args().skip(1) {
            let (key, value) = arg.split_once('=').ok_or(format!("'{arg}' is not key=value"))?;
            let list = |v: &str| v.split(',').map(str::to_owned).collect::<Vec<_>>();
            match key {
                "model" => options.model = value.into(),
                "trace" => options.trace = Some(value.into()),
                "devices" => {
                    options.devices =
                        list(value).iter().map(|d| d.parse()).collect::<Result<_, _>>()?
                }
                "vram-gib" => {
                    options.vram_gib =
                        list(value).iter().map(|d| d.parse()).collect::<Result<_, _>>()?
                }
                "ram-gib" => options.ram_gib = value.parse()?,
                "rows" => options.rows = value.parse()?,
                "cold" => options.cold = value.parse()?,
                "warm" => options.warm = value.parse()?,
                "steps" => options.steps = value.parse()?,
                "lookahead" => options.lookahead = value.parse()?,
                "table" => options.table = value.parse()?,
                "table-recall" => options.table_recall = value.parse()?,
                "dense-mb" => options.dense_mb = value.parse()?,
                "reserve" => options.reserve = value.parse()?,
                "lanes" => options.lanes = value.parse()?,
                "profile" => options.profile = value.parse()?,
                "read-ranks" => options.read_ranks = value.parse()?,
                "name-ranks" => options.name_ranks = value.parse()?,
                // More warm phases after the first, each "<lookahead>:<table layers>".
                "sweep" => {
                    for phase in value.split(',') {
                        let (ahead, table) =
                            phase.split_once(':').ok_or("sweep=<lookahead>:<table>,...")?;
                        options.sweep.push((ahead.parse()?, table.parse()?));
                    }
                }
                "prompt" => options.prompt = value.parse()?,
                "wave-room" => options.wave_room = value.parse()?,
                "share" => options.share = value.parse()?,
                "group" => options.group = value.parse()?,
                "skew" => options.skew = value.parse()?,
                "seed" => options.seed = value.parse()?,
                "sim" => options.sim = value != "0",
                "cpu-bench" => options.cpu_bench = value != "0",
                "serve" => {
                    options.serve = match value {
                        "stream" => Serve::Stream,
                        "cpu" => Serve::Cpu,
                        "model" => Serve::Measured,
                        _ => return Err("serve=stream|cpu|model".into()),
                    }
                }
                "cpus" => {
                    options.cpus =
                        Some(list(value).iter().map(|c| c.parse()).collect::<Result<_, _>>()?)
                }
                "spin-ms" => options.spin_ms = value.parse()?,
                "verify" => options.verify = value != "0",
                "verify-save" => options.verify_save = Some(value.to_owned()),
                "verify-against" => options.verify_against = Some(value.to_owned()),
                "idle-pool" => options.idle_pool = value != "0",
                "alternate" => options.alternate = value.parse()?,
                _ => return Err(format!("unknown option '{key}'").into()),
            }
        }
        if options.model.as_os_str().is_empty() && !options.sim {
            return Err("model=<first part of the model> is needed".into());
        }
        if options.vram_gib.len() != options.devices.len() {
            return Err("vram-gib needs one number per device".into());
        }
        Ok(options)
    }
}

/// A prompt, layer by layer, with the CPU's part: of the experts a layer's
/// rows use and VRAM does not hold, the cost model gives the CPU those with
/// the fewest rows, as many as it computes in the time the link needs for
/// the others. The link's part goes through waves as in [`prompt_pass`]
/// (fetched, not computed: there are no expert kernels for prompt rows yet);
/// the CPU's part is computed for real, for as many (expert, row) pairs as
/// the routing has, from the bytes in the RAM tier, read from the file where
/// they are not there. The rows' activations come down and the CPU's sums
/// go up once per layer, in their real sizes.
fn prompt_pass_hybrid(
    store: &mut ExpertStore<CudaSlots>,
    lanes: &[Lane],
    devices: &[Device],
    shapes: &[ExpertShape],
    trace: &Trace,
    first: usize,
    options: &Options,
) -> Result<(), AnyError> {
    let experts = shapes[0].n_experts as usize;
    let k = shapes[0].top_k as usize;
    let n = shapes[0].n_embd as usize;
    let features = store.cpu_pool().ok_or("no CPU half")?.features();
    // Eight rows of activations stand for the rows of every piece.
    let mut rng = Rng::new(options.seed ^ 0x51ed);
    let x = bench_activations(&mut rng, 8 * n);
    let act8 = quantize_activations(&x, n);
    // What crosses the link per layer for the CPU's part.
    let down_bytes = options.prompt * act_bytes(n, 1);
    let up_bytes = options.prompt * n * 4;
    let mut transfer = Vec::new();
    for device in devices {
        transfer.push((
            DeviceBuffer::new(device, up_bytes as u64)?,
            PinnedBuffer::new(device, up_bytes)?,
            Event::new(device)?,
        ));
    }

    store.take_stats();
    let io_before = store.reader().stats();
    let started = Instant::now();
    let (mut n_waves, mut in_place) = (0usize, 0usize);
    let (mut cpu_experts, mut cpu_pairs, mut link_experts) = (0u64, 0u64, 0u64);
    let mut cpu_busy = Duration::ZERO;
    /// How many experts the CPU has in work at once: their RAM slots are held.
    const WINDOW: usize = 24;
    for (index, lane) in lanes.iter().enumerate() {
        for layer in &lane.layers {
            let shape = &shapes[*layer];
            let mut counts = vec![0u32; experts];
            for token in first..first + options.prompt {
                trace.picks(token, *layer).iter().for_each(|e| counts[*e as usize] += 1);
            }
            // The split of what VRAM does not hold.
            let outside: Vec<u32> = (0..experts as u32)
                .filter(|e| counts[*e as usize] > 0 && !store.in_vram(*layer, *e))
                .collect();
            let misses: Vec<Miss> = outside
                .iter()
                .map(|e| Miss { pairs: counts[*e as usize], heat: 0.0, cpu_only: false })
                .collect();
            let on_cpu = match options.serve {
                Serve::Cpu => vec![true; misses.len()],
                _ => store.cost_model().ok_or("no cost model")?.split(
                    (shape.gate_ty, shape.down_ty),
                    index,
                    &misses,
                    0,
                ),
            };
            let mut link_counts = counts.clone();
            let mut todo: Vec<u32> = Vec::new();
            for (expert, cpu) in outside.iter().zip(&on_cpu) {
                if *cpu {
                    link_counts[*expert as usize] = 0;
                    todo.push(*expert);
                }
            }
            // The CPU's experts: those RAM has first, then by rows.
            todo.sort_by_key(|e| (!store.in_ram(*layer, *e), counts[*e as usize]));
            cpu_experts += todo.len() as u64;
            link_experts += (outside.len() - todo.len()) as u64;

            // The rows' activations come down for the CPU.
            let (device_buffer, host_buffer, sent) = &transfer[index];
            if !todo.is_empty() {
                // SAFETY: both buffers hold `up_bytes`, which is more.
                unsafe {
                    lane.side.copy_to_host(
                        host_buffer.as_ptr().cast_mut(),
                        device_buffer.ptr(),
                        down_bytes,
                    )?;
                }
            }

            let room = ((store.slots(*layer).0 as f64 * options.wave_room) as usize).max(k);
            let plan = store.plan_waves(*layer, &link_counts, room).ok_or("no room for a wave")?;
            in_place += plan.in_place.len();
            let waves: Vec<&[u32]> = plan.waves.iter().map(|w| &w.experts[..]).collect();
            let mut wave = 0;
            let mut wave_running = false;
            let mut next_cpu = 0;
            let mut running: Vec<(u32, std::sync::Arc<LayerJob>)> = Vec::new();
            loop {
                store.poll()?;
                // The link's side: the next wave when the one before is in its slots.
                if wave_running && waves[wave - 1].iter().all(|e| store.in_vram(*layer, *e)) {
                    wave_running = false;
                }
                if !wave_running && wave < waves.len() {
                    let next = waves.get(wave + 1).copied().unwrap_or(&[]);
                    store.wave(*layer, waves[wave], next)?;
                    wave += 1;
                    n_waves += 1;
                    wave_running = true;
                }
                // The CPU's side: jobs that ended give their experts back.
                let mut at = 0;
                while at < running.len() {
                    if running[at].1.is_done() {
                        let (expert, job) = running.swap_remove(at);
                        cpu_busy += job.busy();
                        store.release_from_cpu(*layer, expert);
                    } else {
                        at += 1;
                    }
                }
                // And experts enter the window: computed if RAM has them,
                // asked for if not.
                let window = &todo
                    [next_cpu..(next_cpu + WINDOW - running.len().min(WINDOW)).min(todo.len())];
                store.read_for_cpu(*layer, window);
                while next_cpu < todo.len() && running.len() < WINDOW {
                    let expert = todo[next_cpu];
                    let rows = counts[expert as usize] as usize;
                    let Some(found) = store.hold_for_cpu(*layer, expert, 0xff, [0.1; 8])? else {
                        break;
                    };
                    // One entry per eight rows of the expert, the last with the rest.
                    let pieces: Vec<CpuExpert> = (0..rows.div_ceil(8))
                        .map(|p| CpuExpert { rows: (1u32 << (rows - 8 * p).min(8)) - 1, ..found })
                        .collect();
                    // SAFETY: the matrices are a whole expert of the layer's
                    // shape in the RAM tier, held there until the job has
                    // ended and the hold is released above.
                    let job = std::sync::Arc::new(unsafe {
                        LayerJob::new(features, shape, 8, &act8, &pieces)?
                    });
                    store.cpu_pool().ok_or("no CPU half")?.submit(std::sync::Arc::clone(&job));
                    running.push((expert, job));
                    cpu_pairs += rows as u64;
                    next_cpu += 1;
                }
                if !wave_running
                    && wave == waves.len()
                    && next_cpu == todo.len()
                    && running.is_empty()
                {
                    break;
                }
            }
            if !todo.is_empty() {
                // The CPU's sums for the layer's rows go up.
                // SAFETY: both buffers hold `up_bytes`.
                unsafe {
                    lane.stream.copy_from_host(
                        device_buffer.ptr(),
                        host_buffer.as_ptr(),
                        up_bytes,
                    )?;
                }
                sent.record(&lane.stream)?;
                while !sent.is_done()? {
                    store.poll()?;
                }
            }
            store.record_counts(*layer, &counts, 1.0);
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let stats = store.take_stats();
    let io = store.reader().stats();
    let read = io.demand.device_bytes + io.prefetch.device_bytes
        - io_before.demand.device_bytes
        - io_before.prefetch.device_bytes;
    println!(
        "prompt of {} tokens, layer by layer, with the CPU: {seconds:.2} s, {:.0} tokens/s; {n_waves} waves, {in_place} experts in place, \
         {link_experts} for the link ({} uploaded, {:.2} GiB; {} copied down), {cpu_experts} for the CPU ({cpu_pairs} pairs, workers busy {:.2} s); \
         {} read from the file, {:.2} GiB; load {}",
        options.prompt,
        options.prompt as f64 / seconds,
        stats.uploads,
        gib(stats.upload_bytes),
        stats.demotions,
        cpu_busy.as_secs_f64(),
        stats.demand_reads + stats.guess_reads,
        gib(read),
        load_average(),
    );
    Ok(())
}

/// The three matrices of an expert, read from the file.
fn read_expert(
    map: &ExpertMap,
    reader: &Reader,
    layer: usize,
    expert: usize,
) -> Result<[Vec<u8>; 3], AnyError> {
    use std::os::unix::fs::FileExt;
    let mut parts = [Vec::new(), Vec::new(), Vec::new()];
    let ranges = map.ranges(layer, expert).ok_or("no such expert")?;
    for (part, range) in parts.iter_mut().zip(ranges) {
        let file = std::fs::File::open(&reader.files()[usize::from(range.file.0)].path)?;
        part.resize(range.len, 0);
        file.read_exact_at(part, range.offset)?;
    }
    Ok(parts)
}

/// Adds what a turn counted to the counts of its way of serving.
fn fold(into: &mut StoreStats, turn: StoreStats) {
    into.picks += turn.picks;
    into.hits += turn.hits;
    into.miss_from_ram += turn.miss_from_ram;
    into.miss_from_disk += turn.miss_from_disk;
    into.wait_ms += turn.wait_ms;
    into.demand_read_ms += turn.demand_read_ms;
    into.cpu_experts += turn.cpu_experts;
    into.uploads += turn.uploads;
    into.demotions += turn.demotions;
}

/// Seconds of CPU time this process has used, in user and system mode.
fn process_cpu_seconds() -> f64 {
    // SAFETY: a zeroed `rusage` is valid and the call fills it.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
        seconds(usage.ru_utime) + seconds(usage.ru_stime)
    }
}

fn load_average() -> String {
    let text = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    text.split(' ').next().unwrap_or("?").to_string()
}

/// Rows of activations for measurements: roughly normal with rare large values.
fn bench_activations(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let v: f64 = (0..6).map(|_| rng.unit() - 0.5).sum();
            (if rng.below(61) == 0 { 10.0 * v } else { v }) as f32
        })
        .collect()
}

/// The experts of a job over `experts` (their matrices), every one used by
/// the first `rows` rows.
fn bench_experts(experts: &[&[Vec<u8>; 3]], rows: usize) -> Vec<CpuExpert> {
    experts
        .iter()
        .map(|m| CpuExpert {
            gate: m[0].as_ptr(),
            up: m[1].as_ptr(),
            down: m[2].as_ptr(),
            rows: (1u32 << rows) - 1,
            weight: [0.1; 8],
        })
        .collect()
}

/// The CPU kernels alone, on real expert tensors: their error against exact
/// arithmetic, their time on one core of each kind, and how a job scales
/// over cores. No device is opened.
fn cpu_bench(map: &ExpertMap, reader: &Reader, shapes: &[ExpertShape]) -> Result<(), AnyError> {
    let features = Features::detect()?;
    let topology = Topology::read();
    println!(
        "cpu: x86-64-v3 present, AVX-VNNI {}; performance cores {:?}, their second threads {:?}, efficiency cores {:?}",
        if features.vnni { "yes" } else { "no" },
        topology.performance,
        topology.siblings,
        topology.efficiency
    );
    let mut rng = Rng(0x5EED);
    let pin = |cpu: usize| {
        // SAFETY: a zeroed set is valid and the call only reads it.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
        }
    };

    // One layer of each type: the first that has it.
    let mut layers: Vec<usize> = Vec::new();
    for ty in [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M] {
        layers.extend(shapes.iter().position(|s| s.gate_ty == ty));
    }
    // Enough experts of each that their bytes are not in the cache when a
    // job comes back to one: 40 experts are 50 to 60 MB.
    const POOL: usize = 40;
    let mut matrices: Vec<Vec<[Vec<u8>; 3]>> = Vec::new();
    for layer in &layers {
        matrices.push(
            (0..POOL).map(|e| read_expert(map, reader, *layer, e * 7)).collect::<Result<_, _>>()?,
        );
    }

    // Error against exact arithmetic (f64 on the decoded weights and the
    // unrounded activations), two experts per type.
    for (layer, experts) in layers.iter().zip(&matrices) {
        let shape = &shapes[*layer];
        let (n_in, n_ff) = (shape.n_embd as usize, shape.n_ff as usize);
        for rows in [1usize, 4] {
            let x = bench_activations(&mut rng, rows * n_in);
            let act = quantize_activations(&x, n_in);
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for m in &experts[..2] {
                let job_experts = bench_experts(&[m], rows);
                // SAFETY: the matrices are whole experts of the shape and outlive the job.
                let job = unsafe { LayerJob::new(features, shape, rows, &act, &job_experts)? };
                job.work();
                let sums = job.sums().ok_or("the job did not finish")?;
                let mut w = [
                    vec![0.0f32; n_ff * n_in],
                    vec![0.0f32; n_ff * n_in],
                    vec![0.0f32; n_in * n_ff],
                ];
                ares_quant::dequantize(shape.gate_ty, &m[0], &mut w[0])?;
                ares_quant::dequantize(shape.gate_ty, &m[1], &mut w[1])?;
                ares_quant::dequantize(shape.down_ty, &m[2], &mut w[2])?;
                let dot = |w: &[f32], x: &[f64]| {
                    w.iter().zip(x).map(|(w, x)| f64::from(*w) * x).sum::<f64>()
                };
                for b in 0..rows {
                    let xb: Vec<f64> =
                        x[b * n_in..(b + 1) * n_in].iter().map(|v| f64::from(*v)).collect();
                    let h: Vec<f64> = (0..n_ff)
                        .map(|j| {
                            let g = dot(&w[0][j * n_in..(j + 1) * n_in], &xb);
                            g / (1.0 + (-g).exp()) * dot(&w[1][j * n_in..(j + 1) * n_in], &xb)
                        })
                        .collect();
                    for i in 0..n_in {
                        let want = 0.1 * dot(&w[2][i * n_ff..(i + 1) * n_ff], &h);
                        err += (f64::from(sums[b * n_in + i]) - want).powi(2);
                        norm += want * want;
                    }
                }
            }
            println!(
                "error blk.{layer} {} + {}: {rows} row(s): {:.3} % rms against exact arithmetic",
                shape.gate_ty,
                shape.down_ty,
                100.0 * (err / norm).sqrt()
            );
        }
    }

    // One thread, bound to one core of each kind: microseconds per expert.
    let cores: Vec<(&str, usize)> = [
        topology.performance.get(1).map(|c| ("performance core", *c)),
        topology.efficiency.first().map(|c| ("efficiency core", *c)),
    ]
    .into_iter()
    .flatten()
    .collect();
    for (kind, cpu) in &cores {
        pin(*cpu);
        for (layer, experts) in layers.iter().zip(&matrices) {
            let shape = &shapes[*layer];
            let n_in = shape.n_embd as usize;
            for (rows, vnni, gather) in [
                (1usize, true, true),
                (1, false, true),
                (1, true, false),
                (2, true, true),
                (4, true, true),
                (4, true, false),
                (8, true, true),
            ] {
                if vnni != features.vnni && vnni {
                    continue;
                }
                let features = if vnni { features } else { features.without_vnni() };
                let features =
                    features.with_decode(if gather { Decode::Gather } else { Decode::Loads });
                let x = bench_activations(&mut rng, rows * n_in);
                let act = quantize_activations(&x, n_in);
                let (mut total, mut gate_up) = (Vec::new(), Vec::new());
                for round in 0..3 * POOL {
                    let job_experts = bench_experts(&[&experts[round % POOL]], rows);
                    // SAFETY: as above.
                    let job = unsafe { LayerJob::new(features, shape, rows, &act, &job_experts)? };
                    let start = Instant::now();
                    job.work();
                    total.push(start.elapsed().as_secs_f32() * 1e6);
                    gate_up.push(job.gate_up_elapsed().as_secs_f32() * 1e6);
                }
                let (t, g) = (percentile(&mut total, 0.5), percentile(&mut gate_up, 0.5));
                println!(
                    "one {kind} (cpu {cpu}) blk.{layer} {}: {rows} row(s){}: {t:.0} us per expert                      (gate and up {g:.0}, down {:.0}) = {:.0} us per expert and row, load {}",
                    shape.gate_ty,
                    match (vnni, gather) {
                        (true, true) => "",
                        (false, _) => " without VNNI",
                        (true, false) => " with loads for the gather",
                    },
                    t - g,
                    t / rows as f32,
                    load_average()
                );
            }
        }
    }
    // Back on every CPU the process had.
    // SAFETY: a zeroed set is valid and the call only reads it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for cpu in topology.performance.iter().chain(&topology.siblings).chain(&topology.efficiency)
        {
            libc::CPU_SET(*cpu, &mut set);
        }
        libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
    }

    // Jobs on pools of growing size, the caller spinning on the result as
    // the store's owner does. The caller sits on the first performance core.
    let p = &topology.performance;
    let mut sets: Vec<(String, Vec<usize>)> = Vec::new();
    for n in 1..p.len() {
        sets.push((format!("{n} P"), p[1..=n].to_vec()));
    }
    let all_p: Vec<usize> = p.iter().skip(1).copied().collect();
    if !topology.efficiency.is_empty() {
        let mut cpus = all_p.clone();
        cpus.extend(&topology.efficiency);
        sets.push((format!("{} P + {} E", all_p.len(), topology.efficiency.len()), cpus));
        sets.push((format!("{} E", topology.efficiency.len()), topology.efficiency.clone()));
    }
    if !topology.siblings.is_empty() {
        let mut cpus = all_p.clone();
        cpus.extend(topology.siblings.iter().skip(1));
        sets.push((
            format!("{} P + {} second threads", all_p.len(), topology.siblings.len() - 1),
            cpus.clone(),
        ));
        cpus.extend(&topology.efficiency);
        sets.push((
            format!("{} P + second threads + {} E", all_p.len(), topology.efficiency.len()),
            cpus,
        ));
    }
    if let Some(first) = p.first() {
        pin(*first);
    }
    let (layer, experts) = (layers[0], &matrices[0]);
    let shape = &shapes[layer];
    let n_in = shape.n_embd as usize;
    for (name, cpus) in &sets {
        let pool = CpuPool::new(&Placement::pinned(cpus), Duration::from_millis(50))?;
        if name.ends_with(" E") && name.contains("P +") {
            for (gather, tile_ns) in pool.speeds() {
                println!("  a worker: gather {gather:?}, tile {tile_ns:?} ns");
            }
        }
        for (n_experts, rows) in [(1usize, 1usize), (2, 1), (4, 1), (10, 1), (4, 4)] {
            let x = bench_activations(&mut rng, rows * n_in);
            let act = quantize_activations(&x, n_in);
            let (mut wall, mut busy) = (Vec::new(), 0.0f64);
            let rounds = 60;
            for round in 0..rounds {
                let picked: Vec<&[Vec<u8>; 3]> =
                    (0..n_experts).map(|e| &experts[(round * n_experts + e) % POOL]).collect();
                let job_experts = bench_experts(&picked, rows);
                let start = Instant::now();
                // SAFETY: as above.
                let job = std::sync::Arc::new(unsafe {
                    LayerJob::new(pool.features(), shape, rows, &act, &job_experts)?
                });
                pool.submit(std::sync::Arc::clone(&job));
                while !job.is_done() {
                    std::hint::spin_loop();
                }
                wall.push(start.elapsed().as_secs_f32() * 1e6);
                busy += job.busy().as_secs_f64() * 1e6;
            }
            let median = percentile(&mut wall, 0.5);
            println!(
                "pool {name}: {n_experts} expert(s) x {rows} row(s): {median:.0} us per job (p90 {:.0}) = {:.0} us per                  expert and row; threads busy {:.0} us per job, load {}",
                percentile(&mut wall, 0.9),
                median / (n_experts * rows) as f32,
                busy / rounds as f64,
                load_average()
            );
        }
    }
    Ok(())
}

/// The device side of the layers one device runs.
struct Lane {
    kernels: Kernels,
    experts: ExpertKernels,
    stream: Stream,
    side: Stream,
    /// The model's layers on this device, ascending.
    layers: Vec<usize>,
    routers: DeviceBuffer,
    act: DeviceBuffer,
    real_logits: DeviceBuffer,
    forced: DeviceBuffer,
    forced_host: PinnedBuffer,
    records: DeviceBuffer,
    records_host: PinnedBuffer,
    /// The host's copy of each layer's input rows (codes and scales), for
    /// the experts the CPU computes.
    acts_host: PinnedBuffer,
    act_stride: usize,
    guess_record: DeviceBuffer,
    hidden: DeviceBuffer,
    out: DeviceBuffer,
    dense: Option<(DeviceBuffer, DeviceBuffer, u32)>,
    routed: Vec<Event>,
    copied: Vec<Event>,
    done: Event,
    record_stride: usize,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn percentile(values: &mut [f32], p: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<(), AnyError> {
    let options = Options::parse()?;
    const EXPERTS: usize = 512;
    const LAYERS: usize = 48;
    const K: usize = 10;
    let rows = options.rows as usize;
    let needed = options.profile
        + options.prompt
        + (options.cold + options.warm + options.steps * (1 + options.sweep.len()) + 1) * rows;
    let trace = match &options.trace {
        Some(paths) => Trace::load(paths, LAYERS, K)?,
        None => Trace::synthetic(
            LAYERS,
            EXPERTS,
            K,
            needed.max(4000),
            4,
            options.share,
            options.group,
            options.skew,
            0.65,
            options.seed,
        ),
    };
    println!("trace: {}, {} tokens", trace.origin, trace.len());
    if options.sim {
        let (same, fewest, mean, most) = trace.locality(EXPERTS);
        println!(
            "  {:.1} % of a token's picks were picked by the token before it; a layer uses {fewest} to {most} \
             experts over the trace (mean {mean:.0})",
            100.0 * same
        );
        for layers in [4, 8] {
            let train = trace.len() * 2 / 3;
            let (known, precision, recall) = table_quality(&trace, EXPERTS, layers, train);
            println!(
                "  token table over the first {layers} layers, built from {train} tokens: knows {:.1} % of the later \
                 tokens, precision {:.1} %, recall {:.1} % of all their picks",
                100.0 * known,
                100.0 * precision,
                100.0 * recall
            );
        }
        for slots in [43, 85, 128, 170, 256, 300, 384] {
            println!(
                "  {slots:>3} slots per layer ({:>5} in all): {:.1} % of picks served, steps of 1 row; {:.1} % at {rows} rows",
                slots * LAYERS,
                100.0 * simulate(&trace, EXPERTS, slots, 1, 100.min(trace.len() / 4)),
                100.0
                    * simulate(
                        &trace,
                        EXPERTS,
                        slots.max(rows * K),
                        rows,
                        100.min(trace.len() / 4 / rows)
                    ),
            );
        }
        return Ok(());
    }
    if trace.len() < needed && options.trace.is_some() {
        return Err(format!(
            "the trace has {} tokens; profile, cold, warm and steps need {needed}",
            trace.len()
        )
        .into());
    }
    if trace.len() < needed {
        return Err(format!("the trace has {} tokens; the run needs {needed}", trace.len()).into());
    }

    let gguf = Gguf::open(&options.model)?;
    let map = ExpertMap::discover(&gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"])?;
    let reader = Reader::open_gguf(&gguf, Config::default())?;
    let tensor = |layer: usize, suffix: &str| {
        gguf.tensor(&format!("blk.{layer}.{suffix}.weight"))
            .ok_or(format!("no blk.{layer}.{suffix}"))
    };
    let mut shapes = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let (gate, down) = (tensor(layer, "ffn_gate_exps")?, tensor(layer, "ffn_down_exps")?);
        let shape = ExpertShape {
            n_embd: gate.dims.get(0) as u32,
            n_ff: gate.dims.get(1) as u32,
            n_experts: gate.dims.get(2) as u32,
            top_k: K as u32,
            gate_ty: gate.ty,
            down_ty: down.ty,
        };
        shape.check()?;
        shapes.push(shape);
    }
    if options.cpu_bench {
        return cpu_bench(&map, &reader, &shapes);
    }
    let n_devices = options.devices.len();
    let device_of = |layer: usize| layer * n_devices / LAYERS;
    let geometry = |what: &str| format!("layer geometry: {what}");
    let specs: Vec<LayerSpec> = shapes
        .iter()
        .enumerate()
        .map(|(layer, shape)| {
            Ok(LayerSpec {
                device: device_of(layer),
                offsets: [
                    0,
                    shape.up_offset().ok_or_else(|| geometry("up"))?,
                    shape.down_offset().ok_or_else(|| geometry("down"))?,
                ],
                slot_bytes: shape.slot_bytes().ok_or_else(|| geometry("slot"))?,
            })
        })
        .collect::<Result<_, String>>()?;

    let mut config = StoreConfig::new(
        options.rows,
        K as u32,
        options.vram_gib.iter().map(|g| (g * (1u64 << 30) as f64) as u64).collect(),
        (options.ram_gib * (1u64 << 30) as f64) as u64,
    );
    // Turns of streaming and of the CPU on one store: no slots are kept
    // free, or every turn would begin by emptying or filling them.
    config.vram_reserve = if options.alternate > 0 { 0 } else { options.reserve };
    config.read_lanes = options.lanes;
    // A routing profile, as one would ship with a model: the heat of the
    // first tokens of the trace. The tiers are sized and filled from it.
    let profile = (options.profile > 0).then(|| {
        let mut heat = ExpertHeat::new(LAYERS, EXPERTS, 0.9);
        for token in 0..options.profile {
            for layer in 0..LAYERS {
                let picks: Vec<u32> =
                    trace.picks(token, layer).iter().map(|e| u32::from(*e)).collect();
                heat.record(layer, &picks);
            }
            if token % 32 == 31 {
                heat.decay();
            }
        }
        heat
    });
    let plan = StorePlan::new(config, &map, &reader, &specs, profile.as_ref())?;

    // Devices, ledgers, the store.
    let mut pinned = Ledger::new("pinned RAM", 12 << 30);
    let mut devices = Vec::new();
    let mut ledgers = Vec::new();
    let mut slots = Vec::new();
    for (index, ordinal) in options.devices.iter().enumerate() {
        let device = Device::open(*ordinal, Wait::Spin)?;
        let mut ledger = Ledger::new(format!("CUDA{ordinal}"), device.mem_info()?.free);
        let mut device_slots = CudaSlots::new(
            &device,
            &mut ledger,
            &mut pinned,
            &plan.device_pools(index),
            LAYERS,
            EXPERTS as u32,
        )?;
        if options.serve != Serve::Stream {
            device_slots.add_host_shares(
                &mut ledger,
                &mut pinned,
                shapes[0].host_share_bytes(options.rows),
            )?;
        }
        slots.push(device_slots);
        devices.push(device);
        ledgers.push(ledger);
    }
    let region = HostRegion::pinned(&devices[0], &mut pinned, plan.ram_bytes())?;
    let total_slots: (usize, usize) =
        (0..LAYERS).map(|l| plan.slots(l)).fold((0, 0), |a, s| (a.0 + s.0, a.1 + s.1));
    println!(
        "tiers: {} VRAM slots ({}), {} RAM slots ({:.2} GiB pinned), layer 0 has {:?}, of {} experts",
        total_slots.0,
        (0..n_devices)
            .map(|d| format!("{:.2} GiB", gib(plan.vram_bytes(d))))
            .collect::<Vec<_>>()
            .join(" + "),
        total_slots.1,
        gib(plan.ram_bytes() as u64),
        plan.slots(0),
        LAYERS * EXPERTS,
    );
    let mut store = ExpertStore::new(plan, map, reader, slots, region, profile.as_ref())?;
    let mut workers = 0;
    let _idle_pool = if options.idle_pool && options.serve == Serve::Stream {
        let placement = Placement::for_machine(&Topology::read(), 1);
        println!("cpu: {} idle workers spin next to the step", placement.cpus.len());
        Some(CpuPool::new(&placement, Duration::from_secs(3600))?)
    } else {
        None
    };
    if options.serve != Serve::Stream {
        let topology = Topology::read();
        // The engine's placement, with one core left to this thread (the
        // conductor's part here), unless the run names the CPUs.
        let placement = match &options.cpus {
            Some(cpus) => Placement::pinned(cpus),
            None => Placement::for_machine(&topology, 1),
        };
        let cpus: Vec<String> =
            placement.cpus.iter().map(|c| c.map_or("any".to_owned(), |c| c.to_string())).collect();
        workers = cpus.len();
        let pool = CpuPool::new(&placement, Duration::from_millis(options.spin_ms))?;
        let calibrating = Instant::now();
        store.enable_cpu(pool, shapes.clone(), options.serve)?;
        let model = store.cost_model().ok_or("no cost model")?;
        // The bench's file has down as Q2_0 in every layer.
        let us = |ty: GgmlType, pairs: usize| {
            model.job_time((ty, GgmlType::Q2_0), pairs).map_or(f64::NAN, |t| t * 1e6)
        };
        println!(
            "cpu: {workers} workers on {cpus:?}; calibrated in {:.0} ms: a job of 1 / 2 / 4 / 8 pairs takes {:.0} / {:.0} / {:.0} / {:.0} us (iq2_s), {:.0} / {:.0} / {:.0} / {:.0} (iq2_xxs), {:.0} / {:.0} / {:.0} / {:.0} (iq1_m), load {}",
            calibrating.elapsed().as_secs_f64() * 1e3,
            us(GgmlType::IQ2_S, 1),
            us(GgmlType::IQ2_S, 2),
            us(GgmlType::IQ2_S, 4),
            us(GgmlType::IQ2_S, 8),
            us(GgmlType::IQ2_XXS, 1),
            us(GgmlType::IQ2_XXS, 2),
            us(GgmlType::IQ2_XXS, 4),
            us(GgmlType::IQ2_XXS, 8),
            us(GgmlType::IQ1_M, 1),
            us(GgmlType::IQ1_M, 2),
            us(GgmlType::IQ1_M, 4),
            us(GgmlType::IQ1_M, 8),
            load_average()
        );
    }
    if let Some(heat) = &profile {
        // Fill the tiers with the hottest experts of every layer.
        let started = Instant::now();
        let before = store.reader().stats();
        for layer in 0..LAYERS {
            let mut order: Vec<u32> = (0..EXPERTS as u32).collect();
            order.sort_by(|a, b| {
                heat.layer(layer)[*b as usize].total_cmp(&heat.layer(layer)[*a as usize])
            });
            let (vram, ram) = store.slots(layer);
            order.truncate((vram + ram - options.reserve.min(vram)).min(EXPERTS));
            order.retain(|e| heat.layer(layer)[*e as usize] > 0.0);
            store.fill(layer, &order)?;
        }
        // Until nothing moves any more (what does not fit stays in the file).
        let mut last = (store.stats().uploads, store.stats().demand_reads, Instant::now());
        while !store.is_settled() && last.2.elapsed() < Duration::from_secs(2) {
            store.poll()?;
            let now = (store.stats().uploads, store.stats().demand_reads);
            if now != (last.0, last.1) {
                last = (now.0, now.1, Instant::now());
            }
        }
        let after = store.reader().stats();
        let bytes = after.prefetch.device_bytes + after.demand.device_bytes
            - before.prefetch.device_bytes
            - before.demand.device_bytes;
        let stats = store.take_stats();
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "filled from a profile of {} tokens in {seconds:.1} s: {:.2} GiB read ({:.2} GiB/s), {} experts uploaded",
            options.profile,
            gib(bytes),
            gib(bytes) / seconds,
            stats.uploads
        );
    }

    // The lanes: routers, scratch, events.
    let n = shapes[0].n_embd;
    let mut rng = Rng::new(options.seed ^ 0xabcdef);
    let x: Vec<u8> = (0..rows * n as usize).flat_map(|_| rng.normal().to_le_bytes()).collect();
    let record_bytes = shapes[0].record_bytes(options.rows) as usize;
    let record_stride = record_bytes.next_multiple_of(16);
    let forced_bytes = rows * EXPERTS * 4;
    let act_stride = act_bytes(n as usize, rows).next_multiple_of(16);
    let mut lanes = Vec::new();
    for (index, device) in devices.iter().enumerate() {
        let layers: Vec<usize> = (0..LAYERS).filter(|l| device_of(*l) == index).collect();
        let stream = Stream::new(device, 0)?;
        let kernels = Kernels::load(device)?;
        let router_bytes = shapes[0].router_bytes();
        let routers = DeviceBuffer::new(device, layers.len() as u64 * router_bytes)?;
        for (i, layer) in layers.iter().enumerate() {
            routers.write(i as u64 * router_bytes, gguf.data(tensor(*layer, "ffn_gate_inp")?))?;
        }
        ledgers[index].reserve("routers", layers.len() as u64 * router_bytes)?;
        let xs = DeviceBuffer::new(device, x.len() as u64)?;
        xs.write(0, &x)?;
        let act = DeviceBuffer::new(device, Kernels::quantized_activation_bytes(n, options.rows))?;
        // SAFETY: both buffers were sized for these rows.
        unsafe { kernels.quantize_activations(&stream, xs.ptr(), act.ptr(), n, options.rows)? };
        stream.synchronize()?;
        let dense = if options.dense_mb > 0 {
            // A Q4_K matrix of the size of a layer's dense weights: any bytes
            // with finite scales are one.
            let n_out = (options.dense_mb * 1_000_000 / (u64::from(n) / 256 * 144)) as u32;
            let bytes = u64::from(n_out) * u64::from(n / 256) * 144;
            let w = DeviceBuffer::new(device, bytes)?;
            // SAFETY: the whole buffer.
            unsafe { stream.fill(w.ptr(), 0x11, bytes as usize)? };
            ledgers[index].reserve("dense stand-in", bytes)?;
            Some((
                w,
                DeviceBuffer::new(device, u64::from(options.rows) * u64::from(n_out) * 4)?,
                n_out,
            ))
        } else {
            None
        };
        lanes.push(Lane {
            experts: ExpertKernels::load(device)?,
            side: Stream::new(device, 0)?,
            routers,
            act,
            real_logits: DeviceBuffer::new(device, shapes[0].logits_bytes(options.rows))?,
            forced: DeviceBuffer::new(device, (layers.len() * forced_bytes) as u64)?,
            forced_host: PinnedBuffer::new(device, layers.len() * forced_bytes)?,
            records: DeviceBuffer::new(device, (layers.len() * record_stride) as u64)?,
            records_host: PinnedBuffer::new(device, layers.len() * record_stride)?,
            acts_host: PinnedBuffer::new(device, layers.len() * act_stride)?,
            act_stride,
            guess_record: DeviceBuffer::new(device, record_stride as u64)?,
            hidden: DeviceBuffer::new(device, shapes[0].hidden_bytes(options.rows))?,
            out: DeviceBuffer::new(device, x.len() as u64)?,
            dense,
            routed: (0..layers.len()).map(|_| Event::new(device)).collect::<Result<_, _>>()?,
            copied: (0..layers.len()).map(|_| Event::new(device)).collect::<Result<_, _>>()?,
            done: Event::new(device)?,
            record_stride,
            layers,
            stream,
            kernels,
        });
    }
    for ledger in &ledgers {
        println!(
            "{}: {:.2} GiB reserved for the expert path",
            ledger.name(),
            gib(ledger.reserved())
        );
    }

    let lookahead: Vec<Vec<f64>> = (0..=5).map(lookahead_recall).collect();
    let table_recall = vec![options.table_recall];
    let mut predictions: HashMap<(usize, usize), Vec<u32>> = HashMap::new();

    // The phases: name, steps, and the predictors that run in it.
    let mut phases = vec![
        ("cold".to_owned(), options.cold, options.lookahead, options.table),
        ("warm-up".to_owned(), options.warm, options.lookahead, options.table),
        ("warm".to_owned(), options.steps, options.lookahead, options.table),
    ];
    for (ahead, table) in &options.sweep {
        phases.push((format!("la{ahead} t{table}"), options.steps, (*ahead).min(5), *table));
    }
    let mut token = options.profile;
    let mut step_number = 0u32;
    // The outputs of every step, summed per device in f64, and their bytes.
    let mut checksum: Vec<Vec<f64>> = lanes.iter().map(|_| vec![0.0; rows * n as usize]).collect();
    let mut out_bytes = vec![0u8; rows * n as usize * 4];
    let mut verify_save = match &options.verify_save {
        Some(path) if options.verify => Some(std::io::BufWriter::new(std::fs::File::create(path)?)),
        _ => None,
    };
    let mut verify_against = match &options.verify_against {
        Some(path) if options.verify => Some(std::io::BufReader::new(std::fs::File::open(path)?)),
        _ => None,
    };
    let mut other_bytes = vec![0u8; out_bytes.len()];
    // Per step and device: the relative difference from the other run.
    let mut step_differences: Vec<(f64, u32, usize)> = Vec::new();
    for lane in &lanes {
        // SAFETY: the whole buffer, before any step.
        unsafe { lane.stream.fill(lane.out.ptr(), 0, out_bytes.len())? };
        lane.stream.synchronize()?;
    }
    for (phase, steps, phase_lookahead, phase_table) in phases {
        if steps == 0 {
            continue;
        }
        let (mut named, mut named_right, mut wanted_named) = (0u64, 0u64, 0u64);
        let phase = phase.as_str();
        store.take_stats();
        let io_before = store.reader().stats();
        let mut step_ms: Vec<f32> = Vec::with_capacity(steps);
        let cpu_before = process_cpu_seconds();
        let started = Instant::now();
        // The turns of the measured phase: per way of serving, the steps'
        // times and what the store counted.
        let turns = [Serve::Stream, Serve::Cpu, Serve::Measured];
        let alternating =
            options.alternate > 0 && phase == "warm" && options.serve != Serve::Stream;
        let mut by_turn: Vec<(Vec<f32>, StoreStats)> =
            turns.iter().map(|_| (Vec::new(), StoreStats::default())).collect();
        for step_in_phase in 0..steps {
            if alternating && step_in_phase % options.alternate == 0 {
                let turn = step_in_phase / options.alternate;
                if turn > 0 {
                    fold(&mut by_turn[(turn - 1) % turns.len()].1, store.take_stats());
                }
                store.set_serve(turns[turn % turns.len()]);
            }
            let step_started = Instant::now();
            step_number += 1;
            store.begin_step();

            // Enqueue the whole step on every device.
            for (index, lane) in lanes.iter_mut().enumerate() {
                store.device_mut(index).fence(&lane.stream)?;
                // The picks of the trace as logits: the K picks stand out, in order.
                // SAFETY: the pinned buffer holds a row of logits per layer and row.
                let forced = unsafe {
                    std::slice::from_raw_parts_mut(
                        lane.forced_host.as_mut_ptr().cast::<f32>(),
                        lane.layers.len() * rows * EXPERTS,
                    )
                };
                forced.fill(-30.0);
                for (i, layer) in lane.layers.iter().enumerate() {
                    for row in 0..rows {
                        for (rank, pick) in trace.picks(token + row, *layer).iter().enumerate() {
                            forced[(i * rows + row) * EXPERTS + *pick as usize] =
                                8.0 - 0.5 * rank as f32;
                        }
                    }
                }
                // SAFETY: the buffers have these sizes and live until the step has run.
                unsafe {
                    lane.stream.copy_from_host(
                        lane.forced.ptr(),
                        lane.forced_host.as_ptr(),
                        lane.layers.len() * forced_bytes,
                    )?;
                }
            }
            for (index, lane) in lanes.iter().enumerate() {
                if index > 0 {
                    lane.stream.wait_event(&lanes[index - 1].done)?;
                }
                for (i, layer) in lane.layers.iter().enumerate() {
                    enqueue_layer(
                        &store,
                        lane,
                        index,
                        i,
                        *layer,
                        &shapes,
                        options.rows,
                        step_number,
                        phase_lookahead,
                        options.serve != Serve::Stream,
                    )?;
                }
                lane.done.record(&lane.stream)?;
            }

            // Follow the records while the devices run.
            for lane in &lanes {
                for (i, layer) in lane.layers.iter().enumerate() {
                    while !lane.copied[i].is_done()? {
                        store.poll()?;
                    }
                    // SAFETY: the copy of this record has finished.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            lane.records_host.as_ptr().add(i * lane.record_stride),
                            record_bytes,
                        )
                    };
                    let record = RouteRecord::parse(bytes, K as u32, options.rows)
                        .ok_or("a broken record")?;
                    let groups: Vec<RouteGroup> = record.groups().collect();
                    let picks: Vec<u32> = groups.iter().map(|g| g.expert).collect();
                    debug_assert!((0..rows).all(|row| {
                        trace
                            .picks(token + row, *layer)
                            .iter()
                            .all(|p| picks.contains(&u32::from(*p)))
                    }));
                    if let Some(guess) = predictions.remove(&(step_number as usize, *layer)) {
                        named += guess.len() as u64;
                        named_right += guess.iter().filter(|e| picks.contains(e)).count() as u64;
                        wanted_named += picks.len() as u64;
                    }
                    if options.serve == Serve::Stream {
                        store.routed(*layer, &picks)?;
                    } else if store.serve() == Serve::Stream {
                        // A turn of streaming on a store with a CPU half.
                        store.routed_rows(*layer, record.token(), rows, &groups, &[])?;
                    } else {
                        let n_act = act_bytes(shapes[*layer].n_embd as usize, rows);
                        let act_at = lane.acts_host.as_ptr().wrapping_add(i * lane.act_stride);
                        // SAFETY: the copy of this layer's rows has finished:
                        // it is ahead of the record's event on the side stream.
                        let act = unsafe { std::slice::from_raw_parts(act_at, n_act) };
                        store.routed_rows(*layer, record.token(), rows, &groups, act)?;
                    }
                    // What the routers of later layers say about them, as of now.
                    let ahead = *layer + phase_lookahead;
                    if phase_lookahead > 0 && ahead < LAYERS {
                        let (guess, sure) = named_by(
                            &trace,
                            token,
                            rows,
                            ahead,
                            &lookahead[phase_lookahead],
                            (options.name_ranks, options.read_ranks),
                            &mut rng,
                        );
                        store.predicted(ahead, &guess, sure)?;
                        predictions.insert((step_number as usize, ahead), guess);
                    }
                }
                while !lane.done.is_done()? {
                    store.poll()?;
                }
            }
            store.end_step()?;
            token += rows;

            // Before the next step: what the token table says about its first layers.
            for layer in 0..phase_table.min(LAYERS) {
                let (guess, sure) = named_by(
                    &trace,
                    token,
                    rows,
                    layer,
                    &table_recall,
                    (options.name_ranks, options.read_ranks),
                    &mut rng,
                );
                store.predicted(layer, &guess, sure)?;
                predictions.insert((step_number as usize + 1, layer), guess);
            }
            if phase_table > 0 {
                // The time a token's sampling and the rest of a step's start leave.
                let until = Instant::now() + Duration::from_micros(300);
                while Instant::now() < until {
                    store.poll()?;
                }
            }
            step_ms.push(step_started.elapsed().as_secs_f32() * 1e3);
            if alternating {
                let turn = step_in_phase / options.alternate % turns.len();
                by_turn[turn].0.push(step_started.elapsed().as_secs_f32() * 1e3);
            }
            if options.verify {
                for (index, (lane, sum)) in lanes.iter().zip(&mut checksum).enumerate() {
                    lane.out.read(0, &mut out_bytes)?;
                    for (total, v) in sum.iter_mut().zip(out_bytes.as_chunks::<4>().0.iter()) {
                        *total += f64::from(f32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                    }
                    if let Some(file) = &mut verify_save {
                        std::io::Write::write_all(file, &out_bytes)?;
                    }
                    if let Some(file) = &mut verify_against {
                        std::io::Read::read_exact(file, &mut other_bytes)?;
                        let value = |b: &[u8; 4]| f64::from(f32::from_le_bytes(*b));
                        let (mut diff, mut size) = (0.0f64, 0.0f64);
                        for (ours, theirs) in
                            out_bytes.as_chunks::<4>().0.iter().zip(other_bytes.as_chunks::<4>().0)
                        {
                            diff += (value(ours) - value(theirs)).powi(2);
                            size += value(theirs).powi(2);
                        }
                        step_differences.push((
                            (diff / size.max(1e-30)).sqrt(),
                            step_number,
                            index,
                        ));
                    }
                    // SAFETY: the whole buffer; the step has ended.
                    unsafe { lane.stream.fill(lane.out.ptr(), 0, out_bytes.len())? };
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        if alternating {
            let last = (steps - 1) / options.alternate % turns.len();
            fold(&mut by_turn[last].1, store.take_stats());
            store.set_serve(options.serve);
            for (turn, (ms, stats)) in turns.iter().zip(&mut by_turn) {
                let n = ms.len().max(1) as f64;
                let total: f64 = ms.iter().map(|m| f64::from(*m)).sum();
                println!(
                    "    turn {turn:?}: {} steps, {:.1} steps/s, step median {:.2} ms, p99 {:.2} ms; in a slot {:.2} %; per step: {:.2} from RAM, {:.2} from the file, waiting {:.2} ms of which {:.2} ms reads, {:.2} experts on the CPU, {:.2} uploaded, {:.2} copied down",
                    ms.len(),
                    1e3 * n / total.max(1e-9),
                    percentile(ms, 0.5),
                    percentile(ms, 0.99),
                    100.0 * stats.hits as f64 / stats.picks.max(1) as f64,
                    stats.miss_from_ram as f64 / n,
                    stats.miss_from_disk as f64 / n,
                    stats.wait_ms / n,
                    stats.demand_read_ms / n,
                    stats.cpu_experts as f64 / n,
                    stats.uploads as f64 / n,
                    stats.demotions as f64 / n,
                );
            }
        }
        println!(
            "{phase:>8}: the process kept {:.2} logical CPUs busy (every thread, spinning included), load {}",
            (process_cpu_seconds() - cpu_before) / elapsed,
            load_average()
        );
        let stats = store.take_stats();
        let io = store.reader().stats();
        report(
            phase,
            steps,
            rows,
            elapsed,
            &mut step_ms,
            stats,
            io_before.demand.device_bytes + io_before.prefetch.device_bytes,
            io.demand.device_bytes + io.prefetch.device_bytes,
            workers,
        );
        if let Some(model) = store.cost_model() {
            let us = |t: Option<f64>| t.map_or(f64::NAN, |t| t * 1e6);
            println!(
                "          cost model now: a job of 1 / 2 / 4 pairs {:.0} / {:.0} / {:.0} us (iq2_s); an upload {} us; load {}",
                us(model.job_time((GgmlType::IQ2_S, GgmlType::Q2_0), 1)),
                us(model.job_time((GgmlType::IQ2_S, GgmlType::Q2_0), 2)),
                us(model.job_time((GgmlType::IQ2_S, GgmlType::Q2_0), 4)),
                (0..n_devices)
                    .map(|d| format!("{:.0}", us(model.upload_time(d))))
                    .collect::<Vec<_>>()
                    .join(" / "),
                load_average()
            );
        }
        println!(
            "          the reads themselves (one matrix each, the latest {}): demand median {:.3} ms, p99 {:.3} ms; guesses median {:.3} ms",
            io.demand_latency.samples,
            io.demand_latency.p50.as_secs_f64() * 1e3,
            io.demand_latency.p99.as_secs_f64() * 1e3,
            io.prefetch_latency.p50.as_secs_f64() * 1e3,
        );
        if named > 0 {
            println!(
                "          predictors: {:.1} experts named per step, precision {:.1} %, recall {:.1} % of the picks of the layers they were for",
                named as f64 / steps as f64,
                100.0 * named_right as f64 / named as f64,
                100.0 * named_right as f64 / wanted_named.max(1) as f64
            );
        }
    }
    if let Some(file) = &mut verify_save {
        std::io::Write::flush(file)?;
    }
    if !step_differences.is_empty() {
        // An expert left out or added twice is about a fiftieth of a
        // device's sum over its layers; rounding is five digits below that.
        let mut sorted: Vec<f64> = step_differences.iter().map(|d| d.0).collect();
        sorted.sort_by(f64::total_cmp);
        let worst = step_differences
            .iter()
            .copied()
            .fold((0.0, 0, 0), |worst, d| if d.0 > worst.0 { d } else { worst });
        println!(
            "against the other run, per step and device ({} sums): worst {:.3e} (step {}, device {}), median {:.3e}, 99th percentile {:.3e}, {} above 1e-3",
            sorted.len(),
            worst.0,
            worst.1,
            worst.2,
            sorted[sorted.len() / 2],
            sorted[sorted.len() * 99 / 100],
            sorted.iter().filter(|d| **d > 1e-3).count(),
        );
    }
    if options.verify {
        for (index, sum) in checksum.iter().enumerate() {
            let norm = sum.iter().map(|v| v * v).sum::<f64>().sqrt();
            // A fixed pseudo-random projection: sensitive to where a difference is.
            let projection: f64 = sum
                .iter()
                .enumerate()
                .map(|(i, v)| v * (((i * 2654435761) % 1000) as f64 / 500.0 - 1.0))
                .sum();
            println!(
                "checksum device {index}: norm {norm:.9e}, projection {projection:.9e}, first values {:.6e} {:.6e} {:.6e}",
                sum[0], sum[1], sum[2]
            );
        }
    }
    if options.prompt > 0 && options.serve != Serve::Stream {
        prompt_pass_hybrid(&mut store, &lanes, &devices, &shapes, &trace, token, &options)?;
    } else if options.prompt > 0 {
        prompt_pass(&mut store, &lanes, &shapes, &trace, token, &options)?;
    }
    Ok(())
}

/// Enqueues one layer of the step on its lane: the real router's product,
/// the selection of the trace's picks, the record's way to the host on the
/// side stream, the emulated lookahead, and the experts.
#[allow(clippy::too_many_arguments)]
fn enqueue_layer(
    store: &ExpertStore<CudaSlots>,
    lane: &Lane,
    device: usize,
    i: usize,
    layer: usize,
    shapes: &[ExpertShape],
    rows: u32,
    token: u32,
    lookahead: usize,
    act_always: bool,
) -> Result<(), AnyError> {
    let shape = &shapes[layer];
    let slots = store.device(device);
    let (_, pool) = store.place(layer);
    let router = |i: usize| lane.routers.ptr().offset(i as u64 * shape.router_bytes());
    let forced: DevicePtr =
        lane.forced.ptr().offset((i * rows as usize * shape.n_experts as usize * 4) as u64);
    let record = lane.records.ptr().offset((i * lane.record_stride) as u64);
    // SAFETY: every buffer was sized for these shapes and rows; the tables
    // and pools are the store's, which keeps the rules of the slot table.
    unsafe {
        if let Some((w, y, n_out)) = &lane.dense {
            lane.kernels.matvec(
                &lane.stream,
                GgmlType::Q4_K,
                w.ptr(),
                lane.act.ptr(),
                y.ptr(),
                shape.n_embd,
                *n_out,
                rows,
            )?;
        }
        lane.experts.logits(
            &lane.stream,
            shape,
            router(i),
            lane.act.ptr(),
            lane.real_logits.ptr(),
            rows,
        )?;
        lane.experts.select(
            &lane.stream,
            shape,
            forced,
            slots.table(layer),
            record,
            rows,
            token,
        )?;
        lane.routed[i].record(&lane.stream)?;
        lane.side.wait_event(&lane.routed[i])?;
        lane.side.copy_to_host(
            lane.records_host.as_ptr().add(i * lane.record_stride).cast_mut(),
            record,
            shape.record_bytes(rows) as usize,
        )?;
        if act_always {
            // The layer's input rows as the kernels read them, behind the
            // record: what the CPU needs if the layer misses an expert.
            lane.side.copy_to_host(
                lane.acts_host.as_ptr().add(i * lane.act_stride).cast_mut(),
                lane.act.ptr(),
                act_bytes(shape.n_embd as usize, rows as usize),
            )?;
        }
        lane.copied[i].record(&lane.side)?;
        if lookahead > 0 && i + lookahead < lane.layers.len() {
            // The router of a later layer on this layer's rows: its cost.
            let ahead = lane.layers[i + lookahead];
            lane.experts.route(
                &lane.stream,
                &shapes[ahead],
                router(i + lookahead),
                lane.act.ptr(),
                lane.real_logits.ptr(),
                slots.table(ahead),
                lane.guess_record.ptr(),
                rows,
                token,
            )?;
        }
        lane.experts.experts(
            &lane.stream,
            shape,
            slots.pool(pool).0,
            slots.table(layer),
            record,
            lane.act.ptr(),
            lane.hidden.ptr(),
            lane.out.ptr(),
            slots.host_share(layer),
            rows,
            None,
        )?;
    }
    Ok(())
}

/// A prompt of `options.prompt` tokens, layer by layer: every layer's
/// experts for all rows of the prompt, in waves through the store. The
/// experts of a wave are computed with the decode kernels, eight rows of ten
/// experts per launch, which is what exists for their types today; the
/// launches wait on the device for the wave's experts like a decode step.
fn prompt_pass(
    store: &mut ExpertStore<CudaSlots>,
    lanes: &[Lane],
    shapes: &[ExpertShape],
    trace: &Trace,
    first: usize,
    options: &Options,
) -> Result<(), AnyError> {
    let experts = shapes[0].n_experts as usize;
    let k = shapes[0].top_k as usize;
    let rows = options.rows.max(1) as usize;
    store.take_stats();
    let io_before = store.reader().stats();
    let started = Instant::now();
    let (mut n_waves, mut in_place, mut launches) = (0usize, 0usize, 0u64);
    let mut compute = Duration::ZERO;
    for (index, lane) in lanes.iter().enumerate() {
        for (i, layer) in lane.layers.iter().enumerate() {
            let mut counts = vec![0u32; experts];
            for token in first..first + options.prompt {
                trace.picks(token, *layer).iter().for_each(|e| counts[*e as usize] += 1);
            }
            let room = ((store.slots(*layer).0 as f64 * options.wave_room) as usize).max(k);
            let plan = store.plan_waves(*layer, &counts, room).ok_or("no room for a wave")?;
            in_place += plan.in_place.len();
            // What is in place first, then the waves.
            let mut sets: Vec<&[u32]> = vec![&plan.in_place];
            sets.extend(plan.waves.iter().map(|w| &w.experts[..]));
            for (w, set) in sets.iter().enumerate() {
                if w > 0 {
                    let next = sets.get(w + 1).copied().unwrap_or(&[]);
                    store.wave(*layer, set, next)?;
                    n_waves += 1;
                }
                if rows == 8 {
                    // Ten experts at a time, for eight of their rows per launch.
                    let enqueued = Instant::now();
                    for chunk in set.chunks(k) {
                        if chunk.len() < k {
                            continue;
                        }
                        let most = chunk.iter().map(|e| counts[*e as usize]).max().unwrap_or(0);
                        // SAFETY: the pinned buffer holds a row of logits per layer and row.
                        let forced = unsafe {
                            std::slice::from_raw_parts_mut(
                                lane.forced_host
                                    .as_ptr()
                                    .cast::<f32>()
                                    .cast_mut()
                                    .add(i * rows * experts),
                                rows * experts,
                            )
                        };
                        forced.fill(-30.0);
                        for row in 0..rows {
                            for (rank, e) in chunk.iter().enumerate() {
                                forced[row * experts + *e as usize] = 8.0 - 0.5 * rank as f32;
                            }
                        }
                        let bytes = rows * experts * 4;
                        for _ in 0..most.div_ceil(8) {
                            // SAFETY: as in a decode step; the logits are
                            // copied again for every launch because the
                            // selection overwrites them.
                            unsafe {
                                lane.stream.copy_from_host(
                                    lane.forced.ptr().offset((i * bytes) as u64),
                                    lane.forced_host.as_ptr().add(i * bytes),
                                    bytes,
                                )?;
                            }
                            enqueue_layer(
                                store,
                                lane,
                                index,
                                i,
                                *layer,
                                shapes,
                                options.rows,
                                0,
                                0,
                                false,
                            )?;
                            launches += 1;
                        }
                    }
                    lane.done.record(&lane.stream)?;
                    while !lane.done.is_done()? {
                        store.poll()?;
                    }
                    compute += enqueued.elapsed();
                } else {
                    while store.waiting() > 0 {
                        store.poll()?;
                    }
                }
            }
            store.record_counts(*layer, &counts, 1.0);
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let stats = store.take_stats();
    let io = store.reader().stats();
    let read = io.demand.device_bytes + io.prefetch.device_bytes
        - io_before.demand.device_bytes
        - io_before.prefetch.device_bytes;
    println!(
        "prompt of {} tokens, layer by layer: {seconds:.2} s, {:.0} tokens/s; {n_waves} waves, {in_place} experts in place, \
         {} brought in ({} read from the file, {:.2} GiB; {:.2} GiB uploaded; {} copied down)",
        options.prompt,
        options.prompt as f64 / seconds,
        stats.uploads,
        stats.demand_reads + stats.guess_reads,
        gib(read),
        gib(stats.upload_bytes),
        stats.demotions,
    );
    if launches > 0 {
        println!(
            "          computed with the decode kernels: {launches} launches of 8 rows x 10 experts, {:.2} s of the total with their waits",
            compute.as_secs_f64()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn report(
    phase: &str,
    steps: usize,
    rows: usize,
    elapsed: f64,
    step_ms: &mut [f32],
    mut stats: StoreStats,
    io_before: u64,
    io_after: u64,
    workers: usize,
) {
    let per_step = |n: u64| n as f64 / steps as f64;
    let misses = stats.picks - stats.hits;
    println!(
        "{phase:>8}: {steps} steps of {rows} rows in {elapsed:.2} s: {:.1} steps/s ({:.1} rows/s), step median {:.2} ms, p99 {:.2} ms",
        steps as f64 / elapsed,
        (steps * rows) as f64 / elapsed,
        percentile(step_ms, 0.5),
        percentile(step_ms, 0.99),
    );
    println!(
        "          picks {:.1} per step, in a slot {:.2} %; missing per step: {:.2} from RAM, {:.2} from the file, {:.2} already on their way",
        per_step(stats.picks),
        100.0 * stats.hits as f64 / stats.picks.max(1) as f64,
        per_step(stats.miss_from_ram),
        per_step(stats.miss_from_disk),
        per_step(stats.miss_on_its_way),
    );
    println!(
        "          read from the file per step: {:.2} MB ({:.2} demand reads, {:.2} guesses); uploaded {:.2} MB ({:.2} experts, {:.2} early), copied down {:.2}",
        (io_after - io_before) as f64 / 1e6 / steps as f64,
        per_step(stats.demand_reads),
        per_step(stats.guess_reads),
        stats.upload_bytes as f64 / 1e6 / steps as f64,
        per_step(stats.uploads),
        per_step(stats.early_uploads),
        per_step(stats.demotions),
    );
    println!(
        "          a miss took: median {:.3} ms, p99 {:.3} ms ({misses} misses); from the file: median {:.3} ms, p99 {:.3} ms ({}); from RAM: median {:.3} ms, p99 {:.3} ms ({})",
        percentile(&mut stats.miss_ms, 0.5),
        percentile(&mut stats.miss_ms, 0.99),
        percentile(&mut stats.disk_miss_ms, 0.5),
        percentile(&mut stats.disk_miss_ms, 0.99),
        stats.disk_miss_ms.len(),
        percentile(&mut stats.ram_miss_ms, 0.5),
        percentile(&mut stats.ram_miss_ms, 0.99),
        stats.ram_miss_ms.len(),
    );
    println!(
        "          waiting for misses: {:.2} ms per step (summed over the layers), of which reads of experts {:.2} ms",
        stats.wait_ms / steps as f64,
        stats.demand_read_ms / steps as f64
    );
    if stats.cpu_experts > 0 {
        println!(
            "          on the CPU: {:.2} experts per step ({:.2} pairs) in {:.2} jobs; a layer's sums took median {:.3} ms, p99 {:.3} ms from its record; workers busy {:.2} ms per step = {:.1} % of {workers} workers; promoted {:.2} experts per step",
            per_step(stats.cpu_experts),
            per_step(stats.cpu_pairs),
            per_step(stats.cpu_jobs),
            percentile(&mut stats.cpu_ms, 0.5),
            percentile(&mut stats.cpu_ms, 0.99),
            stats.cpu_busy_ms / steps as f64,
            100.0 * stats.cpu_busy_ms / (elapsed * 1e3 * workers.max(1) as f64),
            per_step(stats.promotions),
        );
    }
}
