//! kvreal: the engine's KV formats and kvarn's quantizer on captured attention data.
//!
//!   kvreal run      --data DIR [--stride 2] [--tmin 512] [--threads 6] [--layers 3,7] --out FILE
//!   kvreal stats    --data DIR
//!   kvreal validate --data DIR --kvarn-ref PATH --work DIR
//!
//! Model dimensions: `--heads 8 --kv 2 --hd 256` (Qwen3.5-0.8B).

mod attn;
mod codecs;
mod data;
mod kvarn;
mod stats;
mod validate;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use attn::{Mask, Sample, Stored, Tail, Task};
use data::{Dims, Layer};
use ares_quant::{KvFormat, KvPolicy};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fam {
    /// The engine's format with the old absolute-maximum scale.
    Absmax,
    /// The same without the group rotation (what a plain `q4_0`-like type is).
    AbsmaxNoRot,
    /// The engine's format as shipped.
    Engine,
    /// kvarn's quantizer.
    Kvarn,
    /// Engine + static per-channel scale from the first 256 cells.
    Scale1,
    /// The same with the square root of the scale.
    Scale05,
    /// Engine + static per-channel mean (cancels in the softmax for K).
    MeanOnly,
    /// Engine + static per-channel mean and scale.
    MeanScale1,
    /// Engine + kvarn's tile normalisation on the original values.
    TileOrig,
    /// Engine + kvarn's tile normalisation after the group rotation.
    TileRot,
    /// Engine + the midrange of each rotated group as an offset (f16 per group).
    GroupMid,
    /// Static mean from the first 128 / 512 cells.
    MeanN128,
    MeanN512,
    /// Static mean over the whole sequence (not causal: an upper bound).
    MeanAll,
    /// Static mean and the square root of the scale.
    MeanScale05,
    /// The mean of every tile of 128 cells (an f16 vector per tile).
    TileMean,
    /// The same with tiles of this many cells.
    TileMeanN(u16),
    /// The mean of the previous tile of this many cells (known when a cell is
    /// written, so cells need not wait for their tile to fill).
    PrevMeanN(u16),
    /// The built codec: `encode_tile`/`decode_tile` with the mean per tile of 128 cells.
    Built,
    /// The stored sequence of `ares_quant::KvHead` under the tile policy,
    /// attended through `KvHead::attend` (the reference of the kernels). It is
    /// not an array: `run_task` feeds a head cell by cell.
    Head,
    /// The row codec (`encode_row`/`decode_row`, no mean): the same arrays as
    /// `Engine`, named separately for the comparison under the tile policy.
    Rows,
}

impl Fam {
    fn name(self) -> String {
        match self {
            Fam::Absmax => "absmax",
            Fam::AbsmaxNoRot => "absmax-norot",
            Fam::Engine => "engine",
            Fam::Kvarn => "kvarn",
            Fam::Scale1 => "engine+scale",
            Fam::Scale05 => "engine+scale^0.5",
            Fam::MeanOnly => "engine+mean",
            Fam::MeanScale1 => "engine+mean+scale",
            Fam::TileOrig => "engine+tiles(orig)",
            Fam::TileRot => "engine+tiles(rot)",
            Fam::GroupMid => "engine+group-offset",
            Fam::MeanN128 => "engine+mean(128)",
            Fam::MeanN512 => "engine+mean(512)",
            Fam::MeanAll => "engine+mean(all)",
            Fam::MeanScale05 => "engine+mean+scale^0.5",
            Fam::TileMean => "engine+tile-mean",
            Fam::TileMeanN(n) => return format!("engine+tile-mean({n})"),
            Fam::PrevMeanN(n) => return format!("engine+prev-mean({n})"),
            Fam::Built => "built",
            Fam::Rows => "rows",
            Fam::Head => "head",
        }
        .to_string()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MaskKind {
    /// Nothing exact.
    None,
    /// The defaults of `KvPolicy::new` as they were when the first measurements
    /// were made (first 128, 64, 32 at q4, q5, q6; newest 256, 128, 64; sliding;
    /// the larger numbers of the two formats), kept here so that the numbers of
    /// `docs/dev/studies/kv-real-data.md` stay reproducible.
    Policy,
    /// `KvPolicy::new(k, v)` of the engine as it is now: whole tiles of 128
    /// between its two windows are stored in the formats, the rest is exact.
    TilePolicy,
    /// First 128 and newest 128 (sliding), kvarn's numbers.
    S128,
    /// Only the group being filled is exact (no first cells): kvarn without its sink.
    Open,
    /// kvarn's own: the first 128 and the open group.
    Kvarn,
    /// The first 128 and the open group and the one before it (129 to 256 newest).
    Kvarn2,
    /// The first `first` cells and the newest `newest` cells (sliding), for the sweep.
    Sweep { first: u16, newest: u16 },
}

impl MaskKind {
    fn name(self) -> String {
        if let MaskKind::Sweep { first, newest } = self {
            return format!("s{first}+{newest}");
        }
        match self {
            MaskKind::None => "none",
            MaskKind::Policy => "policy",
            MaskKind::TilePolicy => "tile-policy",
            MaskKind::S128 => "128/128",
            MaskKind::Open => "open",
            MaskKind::Kvarn => "kvarn",
            MaskKind::Kvarn2 => "kvarn2",
            MaskKind::Sweep { .. } => unreachable!(),
        }
        .to_string()
    }

    fn mask(self, kb: u32, vb: u32) -> Mask {
        match self {
            MaskKind::None => Mask { first: 0, tail: Tail::Sliding(0) },
            MaskKind::Policy => {
                // (first, newest) defaults of the first measurements, per width
                let old = |bits: u32| match bits {
                    8 => (0, 0),
                    7 => (16, 32),
                    6 => (32, 64),
                    5 => (64, 128),
                    _ => (128, 256),
                };
                let (a, b) = (old(kb), old(vb));
                Mask { first: a.0.max(b.0), tail: Tail::Sliding(a.1.max(b.1)) }
            }
            MaskKind::TilePolicy => Mask {
                first: 0,
                tail: Tail::Policy(KvPolicy::new(KvFormat::quantized(kb).unwrap(), KvFormat::quantized(vb).unwrap())),
            },
            MaskKind::S128 => Mask { first: 128, tail: Tail::Sliding(128) },
            MaskKind::Open => Mask { first: 0, tail: Tail::Aligned(1) },
            MaskKind::Kvarn => Mask { first: 128, tail: Tail::Aligned(1) },
            MaskKind::Kvarn2 => Mask { first: 128, tail: Tail::Aligned(2) },
            MaskKind::Sweep { first, newest } => Mask { first: first.into(), tail: Tail::Sliding(newest.into()) },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Spec {
    kfam: Fam,
    vfam: Fam,
    kb: u32,
    vb: u32,
    mask: MaskKind,
}

impl Spec {
    fn dists_wanted(&self) -> bool {
        self.kfam == Fam::Head
    }

    fn name(&self) -> String {
        let fam = if self.kfam == self.vfam {
            self.kfam.name()
        } else {
            format!("K:{}+V:{}", self.kfam.name(), self.vfam.name())
        };
        // the built codec under the tile policy is simply "policy"
        let mask = if (self.kfam == Fam::Built || self.kfam == Fam::Head) && self.vfam == self.kfam && self.mask == MaskKind::TilePolicy {
            "policy".to_string()
        } else {
            self.mask.name()
        };
        format!("{}|K{}V{}|{}", fam, self.kb, self.vb, mask)
    }
}

const PAIRS: [(u32, u32); 6] = [(8, 8), (6, 6), (5, 5), (4, 4), (6, 5), (5, 4)];
const N0: usize = 256;

const PAIRS_BUILT: [(u32, u32); 8] = [(8, 8), (6, 6), (5, 5), (4, 4), (6, 5), (5, 4), (8, 4), (4, 8)];

/// The built codec against the tool's what-if of the same idea, and the engine
/// as it now is: `built|..|kvarn` (kvarn's tail), `built|..|policy` (the tile
/// policy of `KvPolicy`) and `rows|..|tile-policy` (the row codec under the
/// same exact cells).
fn specs_built() -> Vec<Spec> {
    let mut v = Vec::new();
    for (kb, vb) in PAIRS_BUILT {
        if PAIRS.contains(&(kb, vb)) {
            v.push(Spec { kfam: Fam::Built, vfam: Fam::Built, kb, vb, mask: MaskKind::Kvarn });
        }
        v.push(Spec { kfam: Fam::Built, vfam: Fam::Built, kb, vb, mask: MaskKind::TilePolicy });
        v.push(Spec { kfam: Fam::Rows, vfam: Fam::Rows, kb, vb, mask: MaskKind::TilePolicy });
    }
    v
}

/// The built codec with arrays against the same policy through the stored
/// sequence and its reference attention (`KvHead`).
fn specs_head() -> Vec<Spec> {
    let mut v = Vec::new();
    for (kb, vb) in PAIRS_BUILT {
        v.push(Spec { kfam: Fam::Built, vfam: Fam::Built, kb, vb, mask: MaskKind::TilePolicy });
        v.push(Spec { kfam: Fam::Head, vfam: Fam::Head, kb, vb, mask: MaskKind::TilePolicy });
    }
    v
}

fn specs_tilemean() -> Vec<Spec> {
    let mut v = Vec::new();
    for (kb, vb) in PAIRS {
        let mut add = |kfam: Fam, vfam: Fam| v.push(Spec { kfam, vfam, kb, vb, mask: MaskKind::Kvarn });
        add(Fam::Engine, Fam::Engine);
        for n in [32u16, 64, 128, 256, 512] {
            add(Fam::TileMeanN(n), Fam::TileMeanN(n));
        }
        for n in [64u16, 128, 256] {
            add(Fam::PrevMeanN(n), Fam::PrevMeanN(n));
        }
        add(Fam::TileMeanN(128), Fam::Engine);
        add(Fam::Engine, Fam::TileMeanN(128));
        add(Fam::PrevMeanN(128), Fam::Engine);
        add(Fam::Engine, Fam::PrevMeanN(128));
    }
    v
}

fn specs_sweep() -> Vec<Spec> {
    let mut v = Vec::new();
    for (kb, vb) in [(4, 4), (5, 5), (6, 6)] {
        for first in [0u16, 4, 16, 64, 128, 256] {
            for newest in [0u16, 32, 128, 256, 512, 1024] {
                v.push(Spec { kfam: Fam::Engine, vfam: Fam::Engine, kb, vb, mask: MaskKind::Sweep { first, newest } });
            }
        }
    }
    v
}

fn specs_extra() -> Vec<Spec> {
    let mut v = Vec::new();
    for (kb, vb) in PAIRS {
        for fam in [Fam::MeanOnly, Fam::MeanN128, Fam::MeanN512, Fam::MeanAll, Fam::MeanScale05, Fam::TileMean] {
            for mask in [MaskKind::Open, MaskKind::Kvarn] {
                v.push(Spec { kfam: fam, vfam: fam, kb, vb, mask });
            }
        }
        for fam in [Fam::Engine, Fam::Kvarn] {
            for mask in [MaskKind::Open, MaskKind::Kvarn] {
                v.push(Spec { kfam: fam, vfam: fam, kb, vb, mask });
            }
        }
    }
    v
}

fn specs() -> Vec<Spec> {
    let mut v = Vec::new();
    for (kb, vb) in PAIRS {
        let mut add = |kfam: Fam, vfam: Fam, masks: &[MaskKind]| {
            for &mask in masks {
                v.push(Spec { kfam, vfam, kb, vb, mask });
            }
        };
        use MaskKind::*;
        add(Fam::Absmax, Fam::Absmax, &[None, Policy, S128, Open, Kvarn]);
        add(Fam::Engine, Fam::Engine, &[None, Policy, S128, Open, Kvarn, Kvarn2]);
        add(Fam::Kvarn, Fam::Kvarn, &[None, Policy, S128, Open, Kvarn, Kvarn2]);
        add(Fam::AbsmaxNoRot, Fam::AbsmaxNoRot, &[None, Policy]);
        for fam in [Fam::Scale1, Fam::Scale05, Fam::MeanOnly, Fam::MeanScale1, Fam::TileOrig, Fam::TileRot, Fam::GroupMid] {
            add(fam, fam, &[Open, Kvarn]);
        }
        // one side only: which of K and V gains
        for fam in [Fam::Kvarn, Fam::MeanOnly, Fam::Scale05, Fam::TileOrig, Fam::GroupMid] {
            add(fam, Fam::Engine, &[Kvarn]);
            add(Fam::Engine, fam, &[Kvarn]);
        }
    }
    v
}

/// What one task (a KV head of a layer) reports.
struct TaskOut {
    layer: usize,
    samples: Vec<Vec<Sample>>,
    /// (family, bits, is_k) -> (signal, error) sums over the whole groups.
    snr: HashMap<(Fam, u32, bool), (f64, f64)>,
    gt_rows: Vec<(u32, f32, f32, f32)>,
    /// Per spec through a `KvHead`: relative distance of its output from that
    /// of the built codec's arrays, per (head, position).
    dists: Vec<Option<Vec<f32>>>,
}

struct Arrays {
    /// Decoded rows by (family, bits, is_k).
    dec: HashMap<(Fam, u32, bool), Vec<f32>>,
    exact_engine: [Vec<f32>; 2],
    exact_kvarn: [Vec<f32>; 2],
}

fn build_arrays(
    k: &[f32],
    v: &[f32],
    hd: usize,
    fams: &[Fam],
    snr: &mut HashMap<(Fam, u32, bool), (f64, f64)>,
) -> Arrays {
    let rows_full = k.len() / hd / 128 * 128;
    let mut dec = HashMap::new();
    let mut put = |dec: &mut HashMap<_, _>, fam: Fam, bits: u32, is_k: bool, x: &[f32], y: Vec<f32>| {
        let (s, e) = codecs::sums(x, &y, hd, rows_full);
        let acc = snr.entry((fam, bits, is_k)).or_insert((0.0, 0.0));
        acc.0 += s;
        acc.1 += e;
        dec.insert((fam, bits, is_k), y);
    };
    for (is_k, x) in [(true, k), (false, v)] {
        let tiles = fams.contains(&Fam::Kvarn).then(|| codecs::KvarnTiles::new(x, hd, is_k));
        let sc_orig = fams.contains(&Fam::TileOrig).then(|| codecs::tile_scales(x, hd, false));
        let sc_rot = fams.contains(&Fam::TileRot).then(|| codecs::tile_scales(x, hd, true));
        let n_all = x.len() / hd;
        // (family, alpha, mean, scale, rows the statistics come from, first row that uses them)
        let statics: Vec<(Fam, codecs::StaticStats, usize)> = [
            (Fam::Scale1, 1.0, false, true, N0, N0),
            (Fam::Scale05, 0.5, false, true, N0, N0),
            (Fam::MeanOnly, 1.0, true, false, N0, N0),
            (Fam::MeanScale1, 1.0, true, true, N0, N0),
            (Fam::MeanN128, 1.0, true, false, 128, 128),
            (Fam::MeanN512, 1.0, true, false, 512, 512),
            (Fam::MeanAll, 1.0, true, false, n_all, 0),
            (Fam::MeanScale05, 0.5, true, true, N0, N0),
        ]
        .into_iter()
        .filter(|(f, ..)| fams.contains(f))
        .map(|(f, alpha, mean, scale, rows, from)| (f, codecs::static_stats(x, hd, rows, alpha, mean, scale), from))
        .collect();
        for bits in [8u32, 6, 5, 4] {
            for &fam in fams {
                let y = match fam {
                    Fam::Absmax => codecs::absmax(x, hd, bits, true),
                    Fam::AbsmaxNoRot => codecs::absmax(x, hd, bits, false),
                    Fam::Engine => codecs::engine(x, hd, bits),
                    Fam::Kvarn => tiles.as_ref().unwrap().decode(bits),
                    Fam::TileOrig => codecs::engine_tile_norm(x, hd, bits, sc_orig.as_ref().unwrap()),
                    Fam::TileRot => codecs::engine_tile_norm(x, hd, bits, sc_rot.as_ref().unwrap()),
                    Fam::GroupMid => codecs::engine_group_offset(x, hd, bits),
                    Fam::Built => codecs::built(x, hd, bits),
                    Fam::Rows => codecs::engine(x, hd, bits),
                    Fam::Head => unreachable!("a stored sequence is not an array"),
                    Fam::TileMean => codecs::engine_tile_mean(x, hd, bits, 128, false),
                    Fam::TileMeanN(n) => codecs::engine_tile_mean(x, hd, bits, n.into(), false),
                    Fam::PrevMeanN(n) => codecs::engine_tile_mean(x, hd, bits, n.into(), true),
                    f => {
                        let (_, st, from) = statics.iter().find(|(g, ..)| *g == f).unwrap();
                        codecs::engine_static(x, hd, bits, st, *from)
                    }
                };
                put(&mut dec, fam, bits, is_k, x, y);
            }
        }
    }
    Arrays {
        dec,
        exact_engine: [codecs::exact_engine(k), codecs::exact_engine(v)],
        exact_kvarn: [codecs::exact_kvarn(k, hd), codecs::exact_kvarn(v, hd)],
    }
}

fn run_task(layer: &Layer, hk: usize, specs: &[Spec], ts: &[usize]) -> TaskOut {
    let dims = layer.dims;
    let (k, v) = (layer.k_head(hk), layer.v_head(hk));
    let q: Vec<Vec<f32>> = (0..dims.group()).map(|g| layer.q_head(hk * dims.group() + g)).collect();
    let task = Task { hd: dims.hd, n: layer.n, k: &k, v: &v, q, ts: ts.to_vec() };
    let gt = attn::ground_truth(&task);
    let nq = ts.len();
    let mut gt_rows = Vec::new();
    for h in 0..dims.group() {
        for (qi, &t) in ts.iter().enumerate() {
            let i = h * nq + qi;
            gt_rows.push((t as u32, gt.mass_first[i], gt.mass_new[i], gt.entropy[i]));
        }
    }
    let mut fams: Vec<Fam> = specs.iter().flat_map(|s| [s.kfam, s.vfam]).filter(|f| *f != Fam::Head).collect();
    if specs.iter().any(|s| s.kfam == Fam::Head) {
        fams.push(Fam::Built);
    }
    fams.sort_by_key(|f| f.name());
    fams.dedup();
    let mut snr = HashMap::new();
    let arrays = build_arrays(&k, &v, dims.hd, &fams, &mut snr);
    let mut samples = Vec::with_capacity(specs.len());
    let mut dists = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.kfam == Fam::Head {
            // the same policy and the same pair through the built codec's arrays
            let exact = &arrays.exact_engine;
            let st = Stored {
                kq: &arrays.dec[&(Fam::Built, spec.kb, true)],
                ke: &exact[0],
                vq: &arrays.dec[&(Fam::Built, spec.vb, false)],
                ve: &exact[1],
                mask: spec.mask.mask(spec.kb, spec.vb),
            };
            let policy = KvPolicy::new(KvFormat::quantized(spec.kb).unwrap(), KvFormat::quantized(spec.vb).unwrap());
            let (s, d) = evaluate_head(&task, &gt, policy, &st);
            samples.push(s);
            dists.push(Some(d));
            continue;
        }
        dists.push(None);
        let exact = |fam: Fam, side: usize| if fam == Fam::Kvarn { &arrays.exact_kvarn[side] } else { &arrays.exact_engine[side] };
        let st = Stored {
            kq: &arrays.dec[&(spec.kfam, spec.kb, true)],
            ke: exact(spec.kfam, 0),
            vq: &arrays.dec[&(spec.vfam, spec.vb, false)],
            ve: exact(spec.vfam, 1),
            mask: spec.mask.mask(spec.kb, spec.vb),
        };
        samples.push(attn::evaluate(&task, &gt, &st));
    }
    TaskOut { layer: layer.il, samples, snr, gt_rows, dists }
}

/// The queries of a task through a `KvHead` that is fed cell by cell: at a
/// sampled position `t` it holds the `t + 1` cells and `attend` runs over all
/// of them. Returns the samples against the exact attention and, per sample,
/// the relative distance of the output from `stored_output` of the built
/// codec's arrays under the same policy.
fn evaluate_head(task: &Task, gt: &attn::Gt, policy: KvPolicy, built: &Stored) -> (Vec<Sample>, Vec<f32>) {
    let hd = task.hd;
    let nq = task.ts.len();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut head = ares_quant::KvHead::new(policy, hd).unwrap();
    let mut samples = Vec::new();
    let mut dists = Vec::new();
    let mut fed = 0;
    for (qi, &t) in task.ts.iter().enumerate() {
        while fed <= t {
            head.append(&task.k[fed * hd..(fed + 1) * hd], &task.v[fed * hd..(fed + 1) * hd]).unwrap();
            fed += 1;
        }
        for (h, qh) in task.q.iter().enumerate() {
            let q = &qh[t * hd..(t + 1) * hd];
            let got = head.attend(q, scale, &[0..t + 1]).unwrap();
            let idx = h * nq + qi;
            let (s, o) = (&gt.s[idx], &gt.o[idx * hd..(idx + 1) * hd]);
            let (mut sum, mut sum_sq, mut kl) = (0.0f64, 0.0f64, 0.0f64);
            let max = got.scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = got.scores.iter().map(|x| (x - max).exp()).sum();
            for (j, p) in gt.p[idx].iter().enumerate() {
                let d = got.scores[j] - f64::from(s[j]);
                sum += d;
                sum_sq += d * d;
                kl += f64::from(*p) * (-d);
            }
            kl += max + z.ln() - gt.lse[idx];
            let cnt = (t + 1) as f64;
            let var = (sum_sq / cnt - (sum / cnt).powi(2)).max(0.0);
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for c in 0..hd {
                num += (f64::from(got.output[c]) - f64::from(o[c])).powi(2);
                den += f64::from(o[c]).powi(2);
            }
            samples.push(Sample {
                head: h as u8,
                t: t as u32,
                rel: (num / den.max(1e-30)).sqrt() as f32,
                kl: kl.max(0.0) as f32,
                score: var.sqrt() as f32,
            });
            let theirs = attn::stored_output(task, built, h, t);
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for c in 0..hd {
                num += (f64::from(got.output[c]) - f64::from(theirs[c])).powi(2);
                den += f64::from(theirs[c]).powi(2);
            }
            dists.push((num / den.max(1e-30)).sqrt() as f32);
        }
    }
    (samples, dists)
}

fn percentile(v: &mut [f32], p: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len().max(1) as f64
}

pub fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

pub fn dims_from(args: &[String]) -> Dims {
    let get = |n: &str, d: usize| arg(args, n).map_or(d, |v| v.parse().unwrap());
    Dims { n_head: get("--heads", 8), n_kv: get("--kv", 2), hd: get("--hd", 256) }
}

fn cmd_run(args: &[String]) {
    let dir = PathBuf::from(arg(args, "--data").expect("--data"));
    let dims = dims_from(args);
    let stride: usize = arg(args, "--stride").map_or(2, |v| v.parse().unwrap());
    let tmin: usize = arg(args, "--tmin").map_or(512, |v| v.parse().unwrap());
    let threads: usize = arg(args, "--threads").map_or(6, |v| v.parse().unwrap());
    let out_path = arg(args, "--out").expect("--out");
    let layers: Vec<usize> = match arg(args, "--layers") {
        Some(l) => l.split(',').map(|x| x.parse().unwrap()).collect(),
        None => data::list_layers(&dir),
    };
    let segments = data::read_segments(&dir.join("segments.txt"));
    let only: Option<Vec<String>> = arg(args, "--only").map(|s| s.split(',').map(String::from).collect());
    let mut specs = match arg(args, "--set") {
        Some("extra") => specs_extra(),
        Some("sweep") => specs_sweep(),
        Some("tilemean") => specs_tilemean(),
        Some("built") => specs_built(),
        Some("head") => specs_head(),
        _ => specs(),
    };
    if let Some(masks) = arg(args, "--masks") {
        let masks: Vec<&str> = masks.split(',').collect();
        specs.retain(|s| masks.contains(&s.mask.name().as_str()));
    }
    if let Some(pairs) = arg(args, "--pairs") {
        let pairs: Vec<&str> = pairs.split(',').collect();
        specs.retain(|s| pairs.contains(&format!("{}{}", s.kb, s.vb).as_str()));
    }
    if let Some(only) = &only {
        specs.retain(|s| only.iter().any(|o| s.name().contains(o.as_str())));
    }
    eprintln!("{} candidates, layers {:?}, stride {stride}, tmin {tmin}, {threads} threads", specs.len(), layers);

    let loaded: Vec<Layer> = layers.iter().map(|&il| data::load_layer(&dir, il, dims)).collect();
    let n = loaded[0].n;
    let ts: Vec<usize> = (tmin..n).step_by(stride).collect();
    eprintln!("{} tokens, {} sampled queries per head", n, ts.len());

    let tasks: Vec<(usize, usize)> =
        (0..loaded.len()).flat_map(|l| (0..dims.n_kv).map(move |h| (l, h))).collect();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<TaskOut>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= tasks.len() {
                        break;
                    }
                    let (l, hk) = tasks[i];
                    let t0 = std::time::Instant::now();
                    let out = run_task(&loaded[l], hk, &specs, &ts);
                    eprintln!("layer {} kv head {hk} done in {:.1}s", loaded[l].il, t0.elapsed().as_secs_f64());
                    results.lock().unwrap().push(out);
                }
            });
        }
    });
    let results = results.into_inner().unwrap();

    let mut report = String::new();
    let seg_of = |t: u32| -> Option<&str> {
        segments.iter().find(|(_, a, b)| (t as usize) >= *a && (t as usize) < *b).map(|(n, ..)| n.as_str())
    };
    let bins: [(&str, usize, usize); 3] = [("t512-1023", 512, 1024), ("t1024-2047", 1024, 2048), ("t2048+", 2048, usize::MAX)];
    let mut layer_ids: Vec<usize> = layers.clone();
    layer_ids.sort();

    for (si, spec) in specs.iter().enumerate() {
        let mut scopes: Vec<(String, Vec<&Sample>)> = Vec::new();
        let all: Vec<(usize, &Sample)> =
            results.iter().flat_map(|r| r.samples[si].iter().map(move |s| (r.layer, s))).collect();
        scopes.push(("all".into(), all.iter().map(|(_, s)| *s).collect()));
        for &il in &layer_ids {
            scopes.push((format!("L{il:02}"), all.iter().filter(|(l, _)| *l == il).map(|(_, s)| *s).collect()));
        }
        for (name, a, b) in bins {
            scopes.push((name.into(), all.iter().filter(|(_, s)| (s.t as usize) >= a && (s.t as usize) < b).map(|(_, s)| *s).collect()));
        }
        let mut labels: Vec<&str> = segments.iter().map(|(n, ..)| n.as_str()).collect();
        labels.sort();
        labels.dedup();
        for label in labels {
            scopes.push((format!("seg:{label}"), all.iter().filter(|(_, s)| seg_of(s.t) == Some(label)).map(|(_, s)| *s).collect()));
        }
        for (scope, ss) in scopes {
            if ss.is_empty() {
                continue;
            }
            let mut rel: Vec<f32> = ss.iter().map(|s| s.rel).collect();
            let mut kl: Vec<f32> = ss.iter().map(|s| s.kl).collect();
            let mut sc: Vec<f32> = ss.iter().map(|s| s.score).collect();
            let _ = writeln!(
                report,
                "R\t{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}",
                spec.name(),
                scope,
                ss.len(),
                mean(&rel),
                percentile(&mut rel, 0.99),
                mean(&kl),
                percentile(&mut kl, 0.99),
                mean(&sc),
                percentile(&mut sc, 0.99),
            );
        }
    }

    // SNR of the codec alone (whole groups), per layer and pooled
    let mut keys: Vec<(Fam, u32, bool)> = results.iter().flat_map(|r| r.snr.keys().copied()).collect();
    keys.sort_by_key(|(f, b, k)| (f.name(), std::cmp::Reverse(*b), !*k));
    keys.dedup();
    for key in keys {
        let db = |s: f64, e: f64| 10.0 * (s / e.max(1e-300)).log10();
        let (mut s, mut e) = (0.0, 0.0);
        for r in &results {
            if let Some((a, b)) = r.snr.get(&key) {
                s += a;
                e += b;
            }
        }
        let _ = writeln!(report, "S\t{}\t{}\t{}\tall\t{:.3}", key.0.name(), key.1, if key.2 { "K" } else { "V" }, db(s, e));
        for &il in &layer_ids {
            let (mut s, mut e) = (0.0, 0.0);
            for r in results.iter().filter(|r| r.layer == il) {
                if let Some((a, b)) = r.snr.get(&key) {
                    s += a;
                    e += b;
                }
            }
            let _ = writeln!(report, "S\t{}\t{}\t{}\tL{il:02}\t{:.3}", key.0.name(), key.1, if key.2 { "K" } else { "V" }, db(s, e));
        }
    }

    // exact attention: where the mass is
    for &il in &layer_ids {
        let rows: Vec<&(u32, f32, f32, f32)> = results.iter().filter(|r| r.layer == il).flat_map(|r| r.gt_rows.iter()).collect();
        for (scope, a, b) in [("all", 0usize, usize::MAX), ("t512-1023", 512, 1024), ("t1024-2047", 1024, 2048), ("t2048+", 2048, usize::MAX)] {
            let sel: Vec<_> = rows.iter().filter(|r| (r.0 as usize) >= a && (r.0 as usize) < b).collect();
            if sel.is_empty() {
                continue;
            }
            let m = |f: fn(&&&(u32, f32, f32, f32)) -> f32| sel.iter().map(|r| f64::from(f(r))).sum::<f64>() / sel.len() as f64;
            let _ = writeln!(report, "G\tL{il:02}\t{scope}\t{}\t{:.4}\t{:.4}\t{:.4}", sel.len(), m(|r| r.1), m(|r| r.2), m(|r| r.3));
        }
    }
    // the stored sequence against the built codec's arrays, query by query
    for (si, spec) in specs.iter().enumerate() {
        if spec.dists_wanted() {
            let mut d: Vec<f32> = results.iter().flat_map(|r| r.dists[si].iter().flatten().copied()).collect();
            let mean_d = mean(&d);
            let max_d = d.iter().copied().fold(0.0f32, f32::max);
            let p99 = percentile(&mut d, 0.99);
            let _ = writeln!(report, "D\t{}\t{:.6e}\t{:.6e}\t{:.6e}", spec.name(), mean_d, p99, max_d);
        }
    }
    // exact cells per query under the tails of the old sliding policy, the tile
    // policy and kvarn's, averaged over the sampled queries, per pair
    let mut pairs_seen: Vec<(u32, u32)> = Vec::new();
    for spec in &specs {
        if !pairs_seen.contains(&(spec.kb, spec.vb)) {
            pairs_seen.push((spec.kb, spec.vb));
        }
    }
    for (kb, vb) in pairs_seen {
        for kind in [MaskKind::Policy, MaskKind::TilePolicy, MaskKind::Kvarn] {
            let mask = kind.mask(kb, vb);
            let (mut exact, mut cells) = (0.0f64, 0.0f64);
            for &t in &ts {
                let (lo, hi) = mask.range(t);
                exact += (t + 1 - (hi - lo)) as f64;
                cells += (t + 1) as f64;
            }
            let _ = writeln!(report, "E\t{}\tK{}V{}\t{:.2}\t{:.2}", kind.name(), kb, vb, exact / ts.len() as f64, cells / ts.len() as f64);
        }
    }
    std::fs::write(out_path, report).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args),
        Some("stats") => stats::run(&args),
        Some("validate") => validate::run(&args),
        _ => eprintln!("usage: kvreal run|stats|validate ..."),
    }
}
