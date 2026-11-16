//! The table of similar experts, from a calibration: how alike two experts
//! of a layer are is measured on the CPU by their outputs on rows the router
//! sends to the first of them, with the engine's own CPU kernels (the
//! device's arithmetic: int8 rows, the file's quantized weights).

use std::io::{BufWriter, Read as _, Write as _};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ares_cuda::{ExpertShape, RouteRecord};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::profile_key;
use ares_model::qwen4exp::weights::Moe;
use ares_model::qwen4exp::{Config, Weights};
use ares_runtime::expert_store::cpu::{CpuExpert, Features, LayerJob, act_bytes};
use ares_runtime::expert_store::{ExpertNeighbours, Near};

use super::{AnyError, Args, load_average};

const MAGIC: &[u8; 8] = b"ARESCALI";

/// Bytes of one row's activations as the CPU reads them.
fn row_act_bytes(n_embd: usize) -> usize {
    act_bytes(n_embd, 1)
}

/// Writes a calibration: a header, then for every row and every layer the
/// row's activations (one row's layout of `act_bytes`), its picks and their
/// weights.
pub struct CalibrationWriter {
    out: BufWriter<std::fs::File>,
    path: std::path::PathBuf,
    n_layer: usize,
    n_embd: usize,
    k: usize,
    rows: u32,
    /// The rows of the step being written: per row, per layer.
    pending: Vec<Vec<Vec<u8>>>,
}

impl CalibrationWriter {
    pub fn create(path: &Path, c: &Config) -> Result<Self, AnyError> {
        let mut out = BufWriter::new(std::fs::File::create(path)?);
        out.write_all(MAGIC)?;
        for v in [c.n_layer as u32, c.n_embd as u32, c.experts.used as u32, 0u32] {
            out.write_all(&v.to_le_bytes())?;
        }
        Ok(Self {
            out,
            path: path.to_owned(),
            n_layer: c.n_layer,
            n_embd: c.n_embd,
            k: c.experts.used,
            rows: 0,
            pending: Vec::new(),
        })
    }

    /// The record and the activations of `layer` for a step of `n_rows` rows.
    pub fn layer(
        &mut self,
        layer: usize,
        n_rows: usize,
        act: &[u8],
        record: &RouteRecord<'_>,
    ) -> Result<(), AnyError> {
        if self.pending.len() != n_rows {
            self.pending = vec![vec![Vec::new(); self.n_layer]; n_rows];
        }
        let n = self.n_embd;
        let ds_at = (n_rows * n).next_multiple_of(16);
        for (row, pending) in self.pending.iter_mut().enumerate() {
            let mut bytes = vec![0u8; row_act_bytes(n) + 8 * self.k];
            bytes[..n].copy_from_slice(&act[row * n..(row + 1) * n]);
            let one_ds = n.next_multiple_of(16);
            let groups = n / 32 * 8;
            bytes[one_ds..one_ds + groups]
                .copy_from_slice(&act[ds_at + row * groups..ds_at + (row + 1) * groups]);
            let at = row_act_bytes(n);
            for (j, (id, w)) in record.ids(row).zip(record.weights(row)).enumerate() {
                bytes[at + 4 * j..at + 4 * j + 4].copy_from_slice(&id.to_le_bytes());
                bytes[at + 4 * self.k + 4 * j..at + 4 * self.k + 4 * j + 4]
                    .copy_from_slice(&w.to_le_bytes());
            }
            pending[layer] = bytes;
        }
        Ok(())
    }

    pub fn rows_done(&mut self, n_rows: usize) -> Result<(), AnyError> {
        for row in self.pending.drain(..) {
            for layer in row {
                if layer.is_empty() {
                    return Err("a layer of a calibrated row was not written".into());
                }
                self.out.write_all(&layer)?;
            }
        }
        self.rows += n_rows as u32;
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), AnyError> {
        self.out.flush()?;
        drop(self.out);
        let mut file = std::fs::OpenOptions::new().write(true).open(&self.path)?;
        use std::io::Seek as _;
        file.seek(std::io::SeekFrom::Start(20))?;
        file.write_all(&self.rows.to_le_bytes())?;
        println!("calibration: {} rows into {}", self.rows, self.path.display());
        Ok(())
    }
}

/// A calibration read back: per layer the rows' activations, picks and
/// weights.
struct Calibration {
    n_embd: usize,
    k: usize,
    rows: usize,
    /// Per layer: `rows` activations of one row's layout each.
    act: Vec<Vec<u8>>,
    ids: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
}

fn read_calibration(path: &Path, layers: std::ops::Range<usize>) -> Result<Calibration, AnyError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 24 || &bytes[..8] != MAGIC {
        return Err(format!("{} is no calibration", path.display()).into());
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let (n_layer, n_embd, k, rows) = (word(8), word(12), word(16), word(20));
    let per_layer = row_act_bytes(n_embd) + 8 * k;
    if bytes.len() < 24 + rows * n_layer * per_layer {
        return Err(format!("{} is cut short", path.display()).into());
    }
    let mut cal = Calibration {
        n_embd,
        k,
        rows,
        act: vec![Vec::new(); n_layer],
        ids: vec![Vec::new(); n_layer],
        weights: vec![Vec::new(); n_layer],
    };
    for row in 0..rows {
        for layer in layers.clone() {
            let at = 24 + (row * n_layer + layer) * per_layer;
            let a = row_act_bytes(n_embd);
            cal.act[layer].extend_from_slice(&bytes[at..at + a]);
            for j in 0..k {
                cal.ids[layer].push(word(at + a + 4 * j) as u32);
                cal.weights[layer].push(f32::from_le_bytes(
                    bytes[at + a + 4 * k + 4 * j..at + a + 4 * k + 4 * j + 4].try_into().unwrap(),
                ));
            }
        }
    }
    Ok(cal)
}

/// The activations of `rows` (indices into the layer's rows) in the layout of
/// a job of that many rows.
fn gather_act(cal: &Calibration, layer: usize, rows: &[usize]) -> Vec<u8> {
    let n = cal.n_embd;
    let one = row_act_bytes(n);
    let mut act = vec![0u8; act_bytes(n, rows.len())];
    let ds_at = (rows.len() * n).next_multiple_of(16);
    let groups = n / 32 * 8;
    for (b, row) in rows.iter().enumerate() {
        let src = &cal.act[layer][row * one..(row + 1) * one];
        act[b * n..(b + 1) * n].copy_from_slice(&src[..n]);
        let one_ds = n.next_multiple_of(16);
        act[ds_at + b * groups..ds_at + (b + 1) * groups]
            .copy_from_slice(&src[one_ds..one_ds + groups]);
    }
    act
}

/// The dequantized activations of row `row` of `layer`: `d * q` per value.
fn dequant_row(cal: &Calibration, layer: usize, row: usize) -> Vec<f32> {
    let n = cal.n_embd;
    let one = row_act_bytes(n);
    let src = &cal.act[layer][row * one..(row + 1) * one];
    let one_ds = n.next_multiple_of(16);
    (0..n)
        .map(|i| {
            let g = i / 32;
            let d = f32::from_le_bytes(src[one_ds + 8 * g..one_ds + 8 * g + 4].try_into().unwrap());
            d * f32::from(src[i] as i8)
        })
        .collect()
}

fn shape_of(c: &Config, moe: &Moe<'_>) -> ExpertShape {
    ExpertShape {
        n_embd: c.n_embd as u32,
        n_ff: c.experts.ff as u32,
        n_experts: c.experts.count as u32,
        top_k: c.experts.used as u32,
        gate_ty: moe.gate.ty(),
        down_ty: moe.down.ty(),
    }
}

/// The outputs of expert `e` on `rows` (`rows.len() * n_embd` values, row by
/// row), computed on this thread.
fn outputs(
    features: Features,
    shape: &ExpertShape,
    moe: &Moe<'_>,
    cal: &Calibration,
    layer: usize,
    e: usize,
    rows: &[usize],
) -> Result<Vec<f32>, AnyError> {
    let n = cal.n_embd;
    let mut out = Vec::with_capacity(rows.len() * n);
    for chunk in rows.chunks(8) {
        let act = gather_act(cal, layer, chunk);
        let expert = CpuExpert {
            gate: moe.gate.expert(e).bytes().as_ptr(),
            up: moe.up.expert(e).bytes().as_ptr(),
            down: moe.down.expert(e).bytes().as_ptr(),
            rows: (1u32 << chunk.len()) - 1,
            weight: [1.0; 8],
        };
        // SAFETY: the matrices are whole experts of the mapped file, which
        // lives as long as the weights.
        let job = unsafe { LayerJob::new(features, shape, chunk.len(), &act, &[expert])? };
        job.work();
        out.extend_from_slice(job.sums().ok_or("a job that did not end")?);
    }
    Ok(out)
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (ab, aa, bb) = (dot(a, b), dot(a, a), dot(b, b));
    if aa <= 0.0 || bb <= 0.0 { 0.0 } else { ab / (aa * bb).sqrt() }
}

/// What a stand-in gives on the rows of the pick it stands in for: the
/// closeness (the squared cosine of the two outputs, weighted by the pick's
/// weight squared, 0 for outputs that point apart) and the least-squares
/// scale.
fn closeness(fx: &[f32], fy: &[f32], w2: &[f64], n: usize) -> (f64, f64) {
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for (b, w) in w2.iter().enumerate() {
        let (x, y) = (&fx[b * n..(b + 1) * n], &fy[b * n..(b + 1) * n]);
        xy += w * dot(x, y);
        xx += w * dot(x, x);
        yy += w * dot(y, y);
    }
    if xx <= 0.0 || yy <= 0.0 || xy <= 0.0 {
        return (0.0, 1.0);
    }
    (xy * xy / (xx * yy), xy / yy)
}

/// For every expert of a layer: the rows it is measured on (those the router
/// picked it for, the largest weight first, then those whose router score
/// for it is highest) and the squared weight of each.
fn own_rows(
    cal: &Calibration,
    layer: usize,
    logits: &[Vec<f32>],
    n_experts: usize,
    cap: usize,
) -> Vec<(Vec<usize>, Vec<f64>)> {
    let k = cal.k;
    let mut picked: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
    for row in 0..cal.rows {
        for j in 0..k {
            let id = cal.ids[layer][row * k + j] as usize;
            picked[id].push((row, cal.weights[layer][row * k + j]));
        }
    }
    (0..n_experts)
        .map(|e| {
            let mut rows = picked[e].clone();
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            rows.truncate(cap);
            let least = cap.clamp(1, 4);
            if rows.len() < least {
                // Too few: the rows whose router score for the expert is
                // highest, with their probability as the weight.
                let mut by_score: Vec<usize> = (0..cal.rows).collect();
                by_score.sort_by(|a, b| logits[*b][e].total_cmp(&logits[*a][e]));
                for row in by_score {
                    if rows.len() >= least {
                        break;
                    }
                    if rows.iter().all(|r| r.0 != row) {
                        let l = &logits[row];
                        let largest = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let total: f32 = l.iter().map(|v| (v - largest).exp()).sum();
                        rows.push((row, (l[e] - largest).exp() / total));
                    }
                }
            }
            let w2 = rows.iter().map(|r| f64::from(r.1) * f64::from(r.1)).collect();
            (rows.into_iter().map(|r| r.0).collect(), w2)
        })
        .collect()
}

/// Runs `f(i)` for `i` in `0..n` on `threads` threads, results in order.
fn parallel<T: Send>(
    n: usize,
    threads: usize,
    f: impl Fn(usize) -> Result<T, AnyError> + Sync,
) -> Result<Vec<T>, AnyError> {
    let next = AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<Option<Result<T, String>>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = f(i).map_err(|e| e.to_string());
                    *results[i].lock().unwrap() = Some(r);
                }
            });
        }
    });
    results
        .into_iter()
        .map(|m| match m.into_inner().unwrap() {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e.into()),
            None => Err("a task did not run".into()),
        })
        .collect()
}

/// The `n` best of `scores` (higher first), leaving out `skip`.
fn top(scores: &[f64], n: usize, skip: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).filter(|i| *i != skip).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    order.truncate(n);
    order
}

/// A sketch of an expert's weights that keeps inner products: each
/// dequantized value goes into one of `dim` buckets with a sign.
fn weight_sketch(moe: &Moe<'_>, e: usize, dim: usize) -> Vec<f32> {
    let mut sketch = vec![0.0f32; dim];
    let mut index = 0u64;
    for m in [moe.gate.expert(e), moe.up.expert(e), moe.down.expert(e)] {
        let mut row = vec![0.0f32; m.cols()];
        for r in 0..m.rows() {
            m.decode_row(r, &mut row);
            for v in &row {
                let h = index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let bucket = (h >> 40) as usize % dim;
                let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
                sketch[bucket] += sign * v;
                index += 1;
            }
        }
    }
    sketch
}

pub fn near(args: &Args) -> Result<(), AnyError> {
    let gguf = Gguf::open(&args.model)?;
    let c = Config::from_gguf(&gguf)?;
    let w = Weights::load(&gguf, &c)?;
    let path = args.calibration.clone().ok_or("--calibration FILE is needed")?;
    let out = args.out.clone().ok_or("--out DIR is needed")?;
    let (first, end) = args.layers.unwrap_or((0, c.n_layer));
    let mut wanted: Vec<usize> = (first..end).collect();
    for s in &args.study {
        if !wanted.contains(s) {
            wanted.push(*s);
        }
    }
    let span = *wanted.iter().min().unwrap_or(&0)..*wanted.iter().max().unwrap_or(&0) + 1;
    let cal = read_calibration(&path, span)?;
    let features = Features::detect()?;
    let key = profile_key(&gguf, &c);
    let n_experts = c.experts.count;
    let n = c.n_embd;
    // A table being made is kept beside the finished one, so that a run cut
    // short goes on where it stopped.
    let mut table = match ExpertNeighbours::read(&out, &key) {
        Ok(Some(table)) if table.count() == args.count => table,
        _ => ExpertNeighbours::new(key, args.count),
    };
    println!(
        "{} rows of calibration, layers {first}..{end}, {} candidates and {} rows per expert, {} \
         threads (load {})",
        cal.rows,
        args.candidates,
        args.rows,
        args.threads,
        load_average()
    );
    for &layer in &wanted {
        let begun = Instant::now();
        let moe = &w.layers[layer].moe;
        let shape = shape_of(&c, moe);
        // The router's scores of every row, and its rows' cosines.
        let rows_f32: Vec<Vec<f32>> = (0..cal.rows).map(|r| dequant_row(&cal, layer, r)).collect();
        let router: Vec<Vec<f32>> = (0..n_experts)
            .map(|e| {
                let mut row = vec![0.0f32; n];
                moe.router.decode_row(e, &mut row);
                row
            })
            .collect();
        let logits: Vec<Vec<f32>> =
            rows_f32.iter().map(|x| router.iter().map(|r| dot(r, x) as f32).collect()).collect();
        let own = own_rows(&cal, layer, &logits, n_experts, args.rows);
        // Outputs on rows every expert shares: spread over the calibration.
        let probe: Vec<usize> = (0..16).map(|i| i * cal.rows / 16).collect();
        let probe_out = parallel(n_experts, args.threads, |e| {
            outputs(features, &shape, moe, &cal, layer, e, &probe)
        })?;
        let probe_cos = |x: usize, y: usize| cosine(&probe_out[x], &probe_out[y]);
        let router_cos = |x: usize, y: usize| cosine(&router[x], &router[y]);

        let study = args.study.contains(&layer);
        if study {
            study_layer(args, features, &shape, moe, &cal, layer, &own, &probe_cos, &router_cos)?;
        }
        if layer < first || layer >= end {
            continue;
        }
        // Candidates by the two cheap measures, then each measured on the
        // expert's own rows.
        let lists = parallel(n_experts, args.threads, |x| {
            let by_probe: Vec<f64> = (0..n_experts).map(|y| probe_cos(x, y)).collect();
            let by_router: Vec<f64> = (0..n_experts).map(|y| router_cos(x, y)).collect();
            let mut candidates = top(&by_probe, args.candidates, x);
            for y in top(&by_router, args.candidates / 3, x) {
                if !candidates.contains(&y) {
                    candidates.push(y);
                }
            }
            let (rows, w2) = &own[x];
            let fx = outputs(features, &shape, moe, &cal, layer, x, rows)?;
            let mut scored: Vec<(usize, f64, f64)> = candidates
                .into_iter()
                .map(|y| {
                    let fy = outputs(features, &shape, moe, &cal, layer, y, rows)?;
                    let (close, scale) = closeness(&fx, &fy, w2, n);
                    Ok((y, close, scale))
                })
                .collect::<Result<_, AnyError>>()?;
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            Ok(scored)
        })?;
        let mut best = Vec::with_capacity(n_experts);
        for (x, scored) in lists.iter().enumerate() {
            for (slot, (y, close, scale)) in table.of_mut(layer, x).iter_mut().zip(scored) {
                *slot = Near { expert: *y as u16, closeness: *close as f32, scale: *scale as f32 };
            }
            best.push(scored.first().map_or(0.0, |s| s.1));
        }
        best.sort_by(f64::total_cmp);
        let at = |p: f64| best[((best.len() - 1) as f64 * p) as usize];
        println!(
            "layer {layer}: {:.1} s; closeness of the nearest: median {:.3}, 10 % {:.3}, 90 % {:.3}, \
             max {:.3} (load {})",
            begun.elapsed().as_secs_f64(),
            at(0.5),
            at(0.1),
            at(0.9),
            at(1.0),
            load_average()
        );
        table.write(&out)?;
    }
    println!("table: {}", ExpertNeighbours::path(&key, &out).display());
    Ok(())
}

/// How alike two experts are by some measure, higher for more alike.
type Measure<'a> = &'a (dyn Fn(usize, usize) -> f64 + Sync);

/// Which measure finds the expert whose output is nearest, on a sample of
/// the layer's experts: the truth is every other expert measured on the
/// sample's own rows.
#[allow(clippy::too_many_arguments)]
fn study_layer(
    args: &Args,
    features: Features,
    shape: &ExpertShape,
    moe: &Moe<'_>,
    cal: &Calibration,
    layer: usize,
    own: &[(Vec<usize>, Vec<f64>)],
    probe_cos: Measure<'_>,
    router_cos: Measure<'_>,
) -> Result<(), AnyError> {
    let begun = Instant::now();
    let n_experts = shape.n_experts as usize;
    let n = shape.n_embd as usize;
    // The most picked experts and a spread of the others: 64 in all.
    let mut picks = vec![0usize; n_experts];
    for id in &cal.ids[layer] {
        picks[*id as usize] += 1;
    }
    let mut by_picks: Vec<usize> = (0..n_experts).collect();
    by_picks.sort_by(|a, b| picks[*b].cmp(&picks[*a]).then(a.cmp(b)));
    let sample: Vec<usize> = (0..64).map(|i| by_picks[i * n_experts / 64]).collect();
    let sketches = parallel(n_experts, args.threads, |e| Ok(weight_sketch(moe, e, 4096)))?;
    let truth = parallel(sample.len(), args.threads, |i| {
        let x = sample[i];
        let (rows, w2) = &own[x];
        let fx = outputs(features, shape, moe, cal, layer, x, rows)?;
        (0..n_experts)
            .map(|y| {
                if y == x {
                    return Ok(f64::NEG_INFINITY);
                }
                let fy = outputs(features, shape, moe, cal, layer, y, rows)?;
                Ok(closeness(&fx, &fy, w2, n).0)
            })
            .collect::<Result<Vec<f64>, AnyError>>()
    })?;
    let measures: [(&str, Measure<'_>); 3] = [
        ("outputs on shared rows", probe_cos),
        ("router rows", router_cos),
        ("weights", &|x, y| cosine(&sketches[x], &sketches[y])),
    ];
    let best: Vec<f64> = truth.iter().map(|t| t.iter().copied().fold(0.0, f64::max)).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "study of layer {layer} ({:.1} s): nearest by the outputs on its own rows (the table's \
         measure): mean closeness {:.3}, median {:.3}",
        begun.elapsed().as_secs_f64(),
        mean(&best),
        {
            let mut b = best.clone();
            b.sort_by(f64::total_cmp);
            b[b.len() / 2]
        }
    );
    for (name, measure) in measures {
        let mut got = Vec::new();
        let mut recall = 0usize;
        for (i, x) in sample.iter().enumerate() {
            let scores: Vec<f64> = (0..n_experts)
                .map(|y| if y == *x { f64::NEG_INFINITY } else { measure(*x, y) })
                .collect();
            let pick = top(&scores, 1, *x)[0];
            got.push(truth[i][pick]);
            let truth_best = top(&truth[i], 1, *x)[0];
            if top(&scores, args.candidates, *x).contains(&truth_best) {
                recall += 1;
            }
        }
        println!(
            "  by {name}: the closeness of its pick {:.3} (of the best {:.0} %), the nearest among \
             its {} first in {} of {}",
            mean(&got),
            100.0 * mean(&got) / mean(&best).max(1e-12),
            args.candidates,
            recall,
            sample.len()
        );
    }
    Ok(())
}
