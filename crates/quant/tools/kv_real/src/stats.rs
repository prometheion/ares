//! Statistics of the captured data that matter for the design of the formats:
//! outliers and loud channels, offsets common to a group, differences of K and V.

use std::path::PathBuf;

use ares_quant::rotate_groups;

use crate::attn::{self, Task};
use crate::data::{self, Layer};
use crate::{arg, dims_from};

#[derive(Default, Clone, Debug)]
struct Stat {
    rms: f64,
    /// Share of the energy that sits in the per-channel means over all rows.
    mean_energy: f64,
    /// Share of the energy in the 8 and in the 16 loudest channels (of hd).
    top8: f64,
    top16: f64,
    /// Loudest channel rms over the median channel rms.
    max_over_median: f64,
    /// Overlap of the 16 loudest channels of the first 256 rows with those of the rest.
    overlap16: f64,
    /// Correlation of the log channel rms of the first 256 rows and of the rest.
    log_corr: f64,
    /// Row norm of row 0 over the median row norm; coefficient of variation of the row norms.
    first_norm_ratio: f64,
    norm_cv: f64,
    /// Excess kurtosis of all values over the global rms: raw, after the group rotation.
    kurt_raw: f64,
    kurt_rot: f64,
    /// Correlation of a channel's value in one row with the next row (mean over channels,
    /// per-channel mean removed).
    lag1: f64,
    /// Fraction of groups of 32 whose largest value holds more than half of the energy:
    /// raw, rotated.
    dom_raw: f64,
    dom_rot: f64,
    /// Fraction of groups with more than half / a quarter of the energy in the mean
    /// (the rotated value 0), and the mean share.
    dc_half: f64,
    dc_quarter: f64,
    dc_mean: f64,
    /// The same with the per-channel means of the first 256 rows removed first.
    dc_half_centered: f64,
    dc_mean_centered: f64,
    /// The first 64 of the channels are the ones RoPE rotates (n_rot = 64): their
    /// share of the energy, and how many of the 16 loudest channels are among them.
    rope_energy: f64,
    top16_in_rope: f64,
}

fn kurtosis(x: &[f32]) -> f64 {
    let n = x.len() as f64;
    let m2 = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| f64::from(*v).powi(4)).sum::<f64>() / n;
    m4 / (m2 * m2) - 3.0
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt().max(1e-30)
}

fn top_set(rms: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..rms.len()).collect();
    idx.sort_by(|&a, &b| rms[b].partial_cmp(&rms[a]).unwrap());
    idx.truncate(k);
    idx
}

fn group_stats(x: &[f32], hd: usize, centre: Option<&[f32]>) -> (f64, f64, f64, f64, f64, f64) {
    // returns (dom_raw, dom_rot, dc_half, dc_quarter, dc_mean, count)
    let mut y = x.to_vec();
    if let Some(m) = centre {
        for row in y.chunks_exact_mut(hd) {
            for (v, mu) in row.iter_mut().zip(m) {
                *v -= mu;
            }
        }
    }
    let mut rot = y.clone();
    rotate_groups(&mut rot).unwrap();
    let (mut dom_raw, mut dom_rot, mut half, mut quarter, mut dc_sum, mut count) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (g, r) in y.chunks_exact(32).zip(rot.chunks_exact(32)) {
        let e: f64 = g.iter().map(|v| f64::from(*v).powi(2)).sum();
        if e <= 0.0 {
            continue;
        }
        let top = g.iter().map(|v| f64::from(*v).powi(2)).fold(0.0, f64::max);
        let top_rot = r.iter().map(|v| f64::from(*v).powi(2)).fold(0.0, f64::max);
        let dc = f64::from(r[0]).powi(2) / e;
        dom_raw += f64::from(u8::from(top / e > 0.5));
        dom_rot += f64::from(u8::from(top_rot / e > 0.5));
        half += f64::from(u8::from(dc > 0.5));
        quarter += f64::from(u8::from(dc > 0.25));
        dc_sum += dc;
        count += 1.0;
    }
    (dom_raw / count, dom_rot / count, half / count, quarter / count, dc_sum / count, count)
}

fn stat(x: &[f32], hd: usize) -> Stat {
    let n = x.len() / hd;
    let n0 = 256.min(n / 2);
    let mut s = Stat::default();
    let total: f64 = x.iter().map(|v| f64::from(*v).powi(2)).sum();
    s.rms = (total / x.len() as f64).sqrt();

    let mut mean = vec![0.0f64; hd];
    let mut e = vec![0.0f64; hd];
    for row in x.chunks_exact(hd) {
        for c in 0..hd {
            mean[c] += f64::from(row[c]);
            e[c] += f64::from(row[c]).powi(2);
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);
    s.mean_energy = mean.iter().map(|m| m * m).sum::<f64>() * n as f64 / total;
    let rms: Vec<f64> = e.iter().map(|v| (v / n as f64).sqrt()).collect();
    s.rope_energy = e[..64].iter().sum::<f64>() / e.iter().sum::<f64>();
    s.top16_in_rope = top_set(&rms, 16).iter().filter(|c| **c < 64).count() as f64 / 16.0;
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let share = |k: usize| sorted.iter().take(k).map(|v| v * v).sum::<f64>() / sorted.iter().map(|v| v * v).sum::<f64>();
    s.top8 = share(8);
    s.top16 = share(16);
    s.max_over_median = sorted[0] / sorted[hd / 2];

    // persistence between the first 256 rows and the rest
    let chan_rms = |rows: &[f32]| -> Vec<f64> {
        let m = (rows.len() / hd) as f64;
        let mut e = vec![0.0f64; hd];
        for row in rows.chunks_exact(hd) {
            for c in 0..hd {
                e[c] += f64::from(row[c]).powi(2);
            }
        }
        e.iter().map(|v| (v / m).sqrt()).collect()
    };
    let (a, b) = (chan_rms(&x[..n0 * hd]), chan_rms(&x[n0 * hd..]));
    let (ta, tb) = (top_set(&a, 16), top_set(&b, 16));
    s.overlap16 = ta.iter().filter(|c| tb.contains(c)).count() as f64 / 16.0;
    let (la, lb): (Vec<f64>, Vec<f64>) = (a.iter().map(|v| v.max(1e-9).ln()).collect(), b.iter().map(|v| v.max(1e-9).ln()).collect());
    s.log_corr = corr(&la, &lb);

    // row norms
    let norms: Vec<f64> = x.chunks_exact(hd).map(|r| r.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt()).collect();
    let mut sn = norms.clone();
    sn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sn[n / 2];
    s.first_norm_ratio = norms[0] / med;
    let mn = norms.iter().sum::<f64>() / n as f64;
    s.norm_cv = (norms.iter().map(|v| (v - mn).powi(2)).sum::<f64>() / n as f64).sqrt() / mn;

    // kurtosis
    s.kurt_raw = kurtosis(x);
    let mut rot = x.to_vec();
    rotate_groups(&mut rot).unwrap();
    s.kurt_rot = kurtosis(&rot);

    // lag-1 correlation per channel
    let mut acc = 0.0;
    for c in 0..hd {
        let (mut sab, mut saa) = (0.0, 0.0);
        for t in 0..n - 1 {
            let a = f64::from(x[t * hd + c]) - mean[c];
            let b = f64::from(x[(t + 1) * hd + c]) - mean[c];
            sab += a * b;
            saa += a * a;
        }
        acc += sab / saa.max(1e-30);
    }
    s.lag1 = acc / hd as f64;

    // groups
    let (dr, dro, h, q, dm, _) = group_stats(x, hd, None);
    s.dom_raw = dr;
    s.dom_rot = dro;
    s.dc_half = h;
    s.dc_quarter = q;
    s.dc_mean = dm;
    let mu0: Vec<f32> = (0..hd).map(|c| x[..n0 * hd].chunks_exact(hd).map(|r| r[c]).sum::<f32>() / n0 as f32).collect();
    let (_, _, h2, _, dm2, _) = group_stats(x, hd, Some(&mu0));
    s.dc_half_centered = h2;
    s.dc_mean_centered = dm2;
    s
}

fn average(v: &[Stat]) -> Stat {
    let n = v.len() as f64;
    let mut out = Stat::default();
    macro_rules! avg {
        ($($f:ident),*) => { $( out.$f = v.iter().map(|s| s.$f).sum::<f64>() / n; )* };
    }
    avg!(rope_energy, top16_in_rope, rms, mean_energy, top8, top16, max_over_median, overlap16, log_corr, first_norm_ratio, norm_cv, kurt_raw, kurt_rot, lag1, dom_raw, dom_rot, dc_half, dc_quarter, dc_mean, dc_half_centered, dc_mean_centered);
    out
}

/// How much of the variance of the scores over the keys sits in the group
/// means (the rotated value 0 of every group of 32 of q and k), and in the
/// single rotated value with the largest share, averaged over sampled queries.
fn score_dominance(layer: &Layer) -> (f64, f64, f64) {
    let hd = layer.dims.hd;
    let ts: Vec<usize> = (600..layer.n).step_by(61).collect();
    let (mut dc_share, mut top_share, mut cnt) = (0.0, 0.0, 0.0);
    let (mut dc_share_c, mut cnt_c) = (0.0, 0.0);
    for hk in 0..layer.dims.n_kv {
        let mut k = layer.k_head(hk);
        let mu0: Vec<f32> = (0..hd).map(|c| k[..256 * hd].chunks_exact(hd).map(|r| r[c]).sum::<f32>() / 256.0).collect();
        let mut kc = k.clone();
        for row in kc.chunks_exact_mut(hd) {
            for (v, m) in row.iter_mut().zip(&mu0) {
                *v -= m;
            }
        }
        rotate_groups(&mut k).unwrap();
        rotate_groups(&mut kc).unwrap();
        for g in 0..layer.dims.group() {
            let mut q = layer.q_head(hk * layer.dims.group() + g);
            rotate_groups(&mut q).unwrap();
            for &t in &ts {
                let qq = &q[t * hd..(t + 1) * hd];
                let var_of = |vals: &[f64]| -> f64 {
                    let m = vals.iter().sum::<f64>() / vals.len() as f64;
                    vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64
                };
                let scores: Vec<f64> = (0..=t).map(|j| (0..hd).map(|i| f64::from(qq[i] * k[j * hd + i])).sum()).collect();
                let vs = var_of(&scores);
                if vs <= 0.0 {
                    continue;
                }
                // the means of the groups: rotated value 0 of each group
                let dc: Vec<f64> = (0..=t).map(|j| (0..hd / 32).map(|g| f64::from(qq[g * 32] * k[j * hd + g * 32])).sum()).collect();
                dc_share += var_of(&dc) / vs;
                // same with K's channel means (first 256 rows) removed: what a static offset leaves
                let scores_c: Vec<f64> = (0..=t).map(|j| (0..hd).map(|i| f64::from(qq[i] * kc[j * hd + i])).sum()).collect();
                let dc_c: Vec<f64> = (0..=t).map(|j| (0..hd / 32).map(|g| f64::from(qq[g * 32] * kc[j * hd + g * 32])).sum()).collect();
                let vs_c = var_of(&scores_c);
                if vs_c > 0.0 {
                    dc_share_c += var_of(&dc_c) / vs_c;
                    cnt_c += 1.0;
                }
                // the single rotated value with the largest share
                let mut best = 0.0f64;
                for i in 0..hd {
                    let one: Vec<f64> = (0..=t).map(|j| f64::from(qq[i] * k[j * hd + i])).collect();
                    best = best.max(var_of(&one) / vs);
                }
                top_share += best;
                cnt += 1.0;
            }
        }
    }
    (dc_share / cnt, dc_share_c / cnt_c, top_share / cnt)
}

/// Where the attention weight of the exact attention goes, averaged over the
/// sampled queries (position >= 512) and heads of a layer.
fn mass_profile(layer: &Layer) {
    let dims = layer.dims;
    let ts: Vec<usize> = (512.min(layer.n - 1)..layer.n).step_by(8).collect();
    let bounds_first = [1usize, 4, 16, 128];
    let bounds_new = [16usize, 128, 256, 1024];
    let mut first = [0.0f64; 4];
    let mut newest = [0.0f64; 4];
    let mut cnt = 0.0;
    for hk in 0..dims.n_kv {
        let (k, v) = (layer.k_head(hk), layer.v_head(hk));
        let q: Vec<Vec<f32>> = (0..dims.group()).map(|g| layer.q_head(hk * dims.group() + g)).collect();
        let task = Task { hd: dims.hd, n: layer.n, k: &k, v: &v, q, ts: ts.clone() };
        let gt = attn::ground_truth(&task);
        for h in 0..dims.group() {
            for (qi, &t) in ts.iter().enumerate() {
                let p = &gt.p[h * ts.len() + qi];
                for (i, b) in bounds_first.iter().enumerate() {
                    first[i] += p.iter().take(*b).map(|x| f64::from(*x)).sum::<f64>();
                }
                for (i, b) in bounds_new.iter().enumerate() {
                    newest[i] += p.iter().skip((t + 1).saturating_sub(*b)).map(|x| f64::from(*x)).sum::<f64>();
                }
                cnt += 1.0;
            }
        }
    }
    println!(
        "L{:02} attention mass (queries >= 512): first 1 / 4 / 16 / 128 cells {:.3} {:.3} {:.3} {:.3} | newest 16 / 128 / 256 / 1024 cells {:.3} {:.3} {:.3} {:.3}",
        layer.il, first[0] / cnt, first[1] / cnt, first[2] / cnt, first[3] / cnt, newest[0] / cnt, newest[1] / cnt, newest[2] / cnt, newest[3] / cnt
    );
}

pub fn run(args: &[String]) {
    let dir = PathBuf::from(arg(args, "--data").expect("--data"));
    let dims = dims_from(args);
    let layers: Vec<Layer> = data::list_layers(&dir).into_iter().map(|il| data::load_layer(&dir, il, dims)).collect();
    println!("tokens {}, layers {:?}, heads {} / kv {} / dim {}", layers[0].n, layers.iter().map(|l| l.il).collect::<Vec<_>>(), dims.n_head, dims.n_kv, dims.hd);
    let mut all: [Vec<Stat>; 2] = [Vec::new(), Vec::new()];
    for layer in &layers {
        for (side, name) in ["K", "V"].iter().enumerate() {
            let per_head: Vec<Stat> =
                (0..dims.n_kv).map(|hk| stat(&if side == 0 { layer.k_head(hk) } else { layer.v_head(hk) }, dims.hd)).collect();
            let s = average(&per_head);
            all[side].push(s.clone());
            print_stat(&format!("L{:02} {name}", layer.il), &s);
        }
        mass_profile(layer);
        let (dc, dcc, top) = score_dominance(layer);
        println!(
            "L{:02} scores: variance share of the 8 group means {:.3}, with K's static channel means removed {:.3}; of the single largest rotated value {:.3}",
            layer.il, dc, dcc, top
        );
    }
    for (side, name) in ["K", "V"].iter().enumerate() {
        print_stat(&format!("mean {name}"), &average(&all[side]));
    }
}

fn print_stat(label: &str, s: &Stat) {
    println!(
        "{label:8} rms {:6.2} | mean-energy {:5.3} top8 {:5.3} top16 {:5.3} max/med {:6.1} overlap16 {:4.2} logcorr {:5.2} | row0 norm/med {:5.2} norm-cv {:5.3} lag1 {:5.2} | kurt raw {:7.1} rot {:6.2} | groups: dom1 raw {:6.4} rot {:6.4}  dc>.5 {:6.4} dc>.25 {:6.4} dc-mean {:5.3} | centred: dc>.5 {:6.4} dc-mean {:5.3} | rope dims: energy {:5.3} loud16 {:4.2}",
        s.rms, s.mean_energy, s.top8, s.top16, s.max_over_median, s.overlap16, s.log_corr, s.first_norm_ratio, s.norm_cv, s.lag1, s.kurt_raw, s.kurt_rot, s.dom_raw, s.dom_rot, s.dc_half, s.dc_quarter, s.dc_mean, s.dc_half_centered, s.dc_mean_centered, s.rope_energy, s.top16_in_rope
    );
}
