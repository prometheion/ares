//! Reference attention and the error of a candidate.
//!
//! A task is one KV head of one layer with the query heads that share it.
//! For every sampled query position `t` and query head the exact attention is
//! computed over positions `0..=t` with the captured Q, K and V (f32, plain
//! softmax, scaling 1/sqrt(head_dim), the query heads grouped onto the KV head
//! as `h / group`), and then the same with the keys and values a candidate
//! stores: cells in the quantized range read back from the candidate's decoded
//! rows, the others from its exact rows.

use ares_quant::KvPolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tail {
    /// The newest `n` cells (including the one just written) are exact.
    Sliding(usize),
    /// Whole groups of 128: the newest `n` groups are exact, the one being
    /// filled included (kvarn: a group is sealed when its last cell is written).
    Aligned(usize),
    /// The cells that `KvPolicy::quantized_cells(n_cells)` of the engine stores
    /// in the formats (whole tiles between its two windows); `first` is unused.
    Policy(KvPolicy),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mask {
    pub first: usize,
    pub tail: Tail,
}

impl Mask {
    /// The quantized range `[lo, hi)` for a query at position `t`: cells before
    /// `lo` and from `hi` on (up to `t`) are exact.
    pub fn range(&self, t: usize) -> (usize, usize) {
        let n_cells = t + 1;
        if let Tail::Policy(p) = self.tail {
            let r = p.quantized_cells(n_cells);
            return (r.start, r.end);
        }
        let lo = self.first.min(n_cells);
        let hi = match self.tail {
            Tail::Sliding(n) => n_cells.saturating_sub(n),
            Tail::Aligned(g) => 128 * (t / 128 + 1).saturating_sub(g),
            Tail::Policy(_) => unreachable!(),
        };
        (lo, hi.clamp(lo, n_cells))
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 16];
    for (x, y) in a.chunks_exact(16).zip(b.chunks_exact(16)) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum()
}

fn axpy(out: &mut [f32], w: f32, v: &[f32]) {
    for (o, x) in out.iter_mut().zip(v) {
        *o += w * x;
    }
}

pub struct Task<'a> {
    pub hd: usize,
    pub n: usize,
    pub k: &'a [f32],
    pub v: &'a [f32],
    /// Query heads of the group, each `[n][hd]`.
    pub q: Vec<Vec<f32>>,
    /// Sampled query positions.
    pub ts: Vec<usize>,
}

/// The exact attention for every (head, sampled position).
pub struct Gt {
    /// `[head * nq + qi]`: scaled scores over `0..=t`.
    pub s: Vec<Vec<f32>>,
    /// The softmax of `s`.
    pub p: Vec<Vec<f32>>,
    pub lse: Vec<f64>,
    /// Outputs, `[(head * nq + qi) * hd ..]`.
    pub o: Vec<f32>,
    /// Attention mass on the first 128 cells, on the newest 128, and entropy (nats).
    pub mass_first: Vec<f32>,
    pub mass_new: Vec<f32>,
    pub entropy: Vec<f32>,
}

pub fn ground_truth(task: &Task) -> Gt {
    let hd = task.hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let nq = task.ts.len();
    let mut gt = Gt {
        s: Vec::new(),
        p: Vec::new(),
        lse: Vec::new(),
        o: vec![0.0; task.q.len() * nq * hd],
        mass_first: Vec::new(),
        mass_new: Vec::new(),
        entropy: Vec::new(),
    };
    for (h, qh) in task.q.iter().enumerate() {
        for (qi, &t) in task.ts.iter().enumerate() {
            let q = &qh[t * hd..(t + 1) * hd];
            let s: Vec<f32> = (0..=t).map(|j| scale * dot(q, &task.k[j * hd..(j + 1) * hd])).collect();
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().map(|x| f64::from(*x)).sum();
            let p: Vec<f32> = e.iter().map(|x| (f64::from(*x) / z) as f32).collect();
            let o = &mut gt.o[(h * nq + qi) * hd..(h * nq + qi + 1) * hd];
            for (j, w) in p.iter().enumerate() {
                axpy(o, *w, &task.v[j * hd..(j + 1) * hd]);
            }
            gt.lse.push(f64::from(m) + z.ln());
            gt.mass_first.push(p.iter().take(128).sum());
            gt.mass_new.push(p.iter().skip((t + 1).saturating_sub(128)).sum());
            gt.entropy.push(p.iter().map(|w| if *w > 0.0 { -w * w.ln() } else { 0.0 }).sum());
            gt.s.push(s);
            gt.p.push(p);
        }
    }
    gt
}

/// What a candidate stores for one KV head.
pub struct Stored<'a> {
    pub kq: &'a [f32],
    pub ke: &'a [f32],
    pub vq: &'a [f32],
    pub ve: &'a [f32],
    pub mask: Mask,
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub head: u8,
    pub t: u32,
    /// Relative L2 error of the attention output.
    pub rel: f32,
    /// KL divergence of the attention weights, exact against candidate (nats).
    pub kl: f32,
    /// Standard deviation over the keys of the error of the scaled scores (nats).
    pub score: f32,
}

/// The output row of query `t` of query head `h` over the stored rows of a
/// candidate, computed as `evaluate` does (`f32` dot products and softmax).
pub fn stored_output(task: &Task, st: &Stored, h: usize, t: usize) -> Vec<f32> {
    let hd = task.hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let q = &task.q[h][t * hd..(t + 1) * hd];
    let (lo, hi) = st.mask.range(t);
    let mut e = vec![0.0f32; t + 1];
    let mut s = vec![0.0f32; t + 1];
    for j in 0..=t {
        let row = if j >= lo && j < hi { st.kq } else { st.ke };
        s[j] = scale * dot(q, &row[j * hd..(j + 1) * hd]);
    }
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f64;
    for j in 0..=t {
        e[j] = (s[j] - m).exp();
        z += f64::from(e[j]);
    }
    let inv = (1.0 / z) as f32;
    let mut o = vec![0.0f32; hd];
    for j in 0..=t {
        let row = if j >= lo && j < hi { st.vq } else { st.ve };
        axpy(&mut o, e[j] * inv, &row[j * hd..(j + 1) * hd]);
    }
    o
}

pub fn evaluate(task: &Task, gt: &Gt, st: &Stored) -> Vec<Sample> {
    let hd = task.hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let nq = task.ts.len();
    let mut samples = Vec::with_capacity(task.q.len() * nq);
    let mut s_hat = vec![0.0f32; task.n];
    let mut e_hat = vec![0.0f32; task.n];
    let mut o_hat = vec![0.0f32; hd];
    for (h, qh) in task.q.iter().enumerate() {
        for (qi, &t) in task.ts.iter().enumerate() {
            let idx = h * nq + qi;
            let q = &qh[t * hd..(t + 1) * hd];
            let (lo, hi) = st.mask.range(t);
            for j in 0..=t {
                let row = if j >= lo && j < hi { st.kq } else { st.ke };
                s_hat[j] = scale * dot(q, &row[j * hd..(j + 1) * hd]);
            }
            let m = s_hat[..=t].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0f64;
            for j in 0..=t {
                e_hat[j] = (s_hat[j] - m).exp();
                z += f64::from(e_hat[j]);
            }
            let lse_hat = f64::from(m) + z.ln();
            let inv = (1.0 / z) as f32;

            o_hat.iter_mut().for_each(|x| *x = 0.0);
            for j in 0..=t {
                let row = if j >= lo && j < hi { st.vq } else { st.ve };
                axpy(&mut o_hat, e_hat[j] * inv, &row[j * hd..(j + 1) * hd]);
            }

            let s = &gt.s[idx];
            let p = &gt.p[idx];
            let (mut kl, mut sum, mut sum_sq) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..=t {
                let d = f64::from(s_hat[j] - s[j]);
                sum += d;
                sum_sq += d * d;
                kl += f64::from(p[j]) * (-d);
            }
            kl += lse_hat - gt.lse[idx];
            let cnt = (t + 1) as f64;
            let var = (sum_sq / cnt - (sum / cnt).powi(2)).max(0.0);

            let o = &gt.o[idx * hd..(idx + 1) * hd];
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for c in 0..hd {
                num += f64::from(o_hat[c] - o[c]).powi(2);
                den += f64::from(o[c]).powi(2);
            }
            samples.push(Sample {
                head: h as u8,
                t: t as u32,
                rel: (num / den.max(1e-30)).sqrt() as f32,
                kl: kl.max(0.0) as f32,
                score: var.sqrt() as f32,
            });
        }
    }
    samples
}
