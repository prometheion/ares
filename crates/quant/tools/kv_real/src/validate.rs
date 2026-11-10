//! Checks of the tool itself:
//! 1. the captured Q, K, V reproduce the attention output the model computed
//!    (and with which grouping of query heads onto KV heads),
//! 2. the absolute-maximum baseline at 8 bits is the engine's `q8` bit for bit,
//! 3. the re-implementation of kvarn's quantizer agrees with the original
//!    routines compiled from beellama.cpp (`kvarn_ref`): head transform, codes,
//!    f16 side vectors and dequantised values.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::attn::{self, Task};
use crate::codecs;
use crate::data::{self, Layer};
use crate::kvarn::{self, T};
use crate::{arg, dims_from};

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_f32s(buf: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

fn read_f32s(path: &PathBuf) -> Vec<f32> {
    fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn attention_vs_model(layers: &[Layer]) {
    println!("== captured Q, K, V against the model's own attention output (attn_pregate)");
    for layer in layers {
        let Some(model) = &layer.attn else { continue };
        let dims = layer.dims;
        let ts: Vec<usize> = (300..layer.n).step_by(97).collect();
        let mut rel_group = Vec::new();
        let mut rel_mod = Vec::new();
        for hk in 0..dims.n_kv {
            let (k, v) = (layer.k_head(hk), layer.v_head(hk));
            let q: Vec<Vec<f32>> = (0..dims.group()).map(|g| layer.q_head(hk * dims.group() + g)).collect();
            let task = Task { hd: dims.hd, n: layer.n, k: &k, v: &v, q, ts: ts.clone() };
            let gt = attn::ground_truth(&task);
            for g in 0..dims.group() {
                let h = hk * dims.group() + g;
                for (qi, &t) in ts.iter().enumerate() {
                    let mine = &gt.o[(g * ts.len() + qi) * dims.hd..(g * ts.len() + qi + 1) * dims.hd];
                    let theirs = &model[t * dims.n_head * dims.hd + h * dims.hd..][..dims.hd];
                    rel_group.push(rel(mine, theirs));
                }
            }
        }
        // the wrong grouping (h % n_kv) for comparison
        for hk in 0..dims.n_kv {
            let (k, v) = (layer.k_head(hk), layer.v_head(hk));
            let heads: Vec<usize> = (0..dims.n_head).filter(|h| h % dims.n_kv == hk).collect();
            let q: Vec<Vec<f32>> = heads.iter().map(|&h| layer.q_head(h)).collect();
            let task = Task { hd: dims.hd, n: layer.n, k: &k, v: &v, q, ts: ts.clone() };
            let gt = attn::ground_truth(&task);
            for (g, &h) in heads.iter().enumerate() {
                for (qi, &t) in ts.iter().enumerate() {
                    let mine = &gt.o[(g * ts.len() + qi) * dims.hd..(g * ts.len() + qi + 1) * dims.hd];
                    let theirs = &model[t * dims.n_head * dims.hd + h * dims.hd..][..dims.hd];
                    rel_mod.push(rel(mine, theirs));
                }
            }
        }
        let m = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let mx = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
        println!(
            "layer {:2}: {} queries x heads; relative L2 difference to the model: grouping h/{} mean {:.2e} max {:.2e}; grouping h%{} mean {:.2e}",
            layer.il,
            rel_group.len(),
            dims.group(),
            m(&rel_group),
            mx(&rel_group),
            dims.n_kv,
            m(&rel_mod)
        );
    }
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}

fn absmax_is_q8(layers: &[Layer]) {
    println!("== absolute-maximum baseline at 8 bits against the engine's q8");
    let layer = &layers[0];
    let hd = layer.dims.hd;
    for (name, x) in [("K", layer.k_head(0)), ("V", layer.v_head(0))] {
        let a = codecs::absmax(&x, hd, 8, true);
        let b = codecs::engine(&x, hd, 8);
        let same = a.iter().zip(&b).filter(|(p, q)| p.to_bits() == q.to_bits()).count();
        println!("layer {} {name}: {} of {} values bit-identical", layer.il, same, a.len());
    }
}

fn unpack(bytes: &[u8], n: usize, bits: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let mut v = 0u8;
            for b in 0..bits {
                let at = i * bits + b;
                v |= ((bytes[at / 8] >> (at % 8)) & 1) << b;
            }
            v
        })
        .collect()
}

fn kvarn_against_original(layers: &[Layer], reference: &str, work: &PathBuf) {
    println!("== re-implementation of kvarn against the original routines of beellama.cpp (src/llama-kvarn.cpp, compiled as is)");
    fs::create_dir_all(work).unwrap();
    let layer = &layers[layers.len() / 2];
    let hd = layer.dims.hd;

    // head transform on real rows
    let x = layer.k_head(0);
    let rows: Vec<&[f32]> = x.chunks_exact(hd).skip(1000).take(64).collect();
    let mut input = Vec::new();
    write_u32(&mut input, rows.len() as u32);
    for r in &rows {
        write_f32s(&mut input, r);
    }
    fs::write(work.join("wht_in.bin"), &input).unwrap();
    let status = Command::new(reference).args(["wht", work.join("wht_in.bin").to_str().unwrap(), work.join("wht_out.f32").to_str().unwrap()]).status().unwrap();
    assert!(status.success());
    let theirs = read_f32s(&work.join("wht_out.f32"));
    let mut same = 0;
    let mut maxdiff = 0.0f32;
    for (i, r) in rows.iter().enumerate() {
        let mut m = r.to_vec();
        kvarn::rotate_head(&mut m);
        for (a, b) in m.iter().zip(&theirs[i * hd..(i + 1) * hd]) {
            same += usize::from(a.to_bits() == b.to_bits());
            maxdiff = maxdiff.max((a - b).abs());
        }
    }
    println!("head transform (256), {} real K rows: {} of {} values bit-identical, max abs difference {:.1e}", rows.len(), same, rows.len() * hd, maxdiff);

    // tiles
    #[derive(Clone)]
    struct Case {
        is_k: bool,
        bits: u32,
        tile: Vec<f32>,
    }
    let mut cases = Vec::new();
    for l in [0, layers.len() / 2, layers.len() - 1] {
        let layer = &layers[l];
        for hk in 0..layer.dims.n_kv {
            for (is_k, x) in [(true, layer.k_head(hk)), (false, layer.v_head(hk))] {
                let mut rot = Vec::new();
                for row in x.chunks_exact(hd) {
                    let mut r = row.to_vec();
                    kvarn::rotate_head(&mut r);
                    rot.extend(r.iter().map(|v| ares_quant::f16_to_f32(ares_quant::f32_to_f16(*v))));
                }
                for g in [0usize, 5, 17, 26] {
                    for s in 0..hd / T {
                        let tile = codecs::KvarnTiles::tile_of(&rot, hd, is_k, g, s);
                        for bits in [4u32, 5, 6, 8] {
                            cases.push(Case { is_k, bits, tile: tile.clone() });
                        }
                    }
                }
            }
        }
    }
    let mut input = Vec::new();
    write_u32(&mut input, cases.len() as u32);
    for c in &cases {
        write_u32(&mut input, u32::from(!c.is_k));
        write_u32(&mut input, c.bits);
        write_f32s(&mut input, &c.tile);
    }
    fs::write(work.join("tiles_in.bin"), &input).unwrap();
    let status = Command::new(reference)
        .args(["tiles", work.join("tiles_in.bin").to_str().unwrap(), work.join("tiles_deq.f32").to_str().unwrap(), work.join("tiles_rec.bin").to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());
    let deq = read_f32s(&work.join("tiles_deq.f32"));
    let rec = fs::read(work.join("tiles_rec.bin")).unwrap();

    struct Agg {
        tiles: usize,
        tiles_identical_codes: usize,
        codes: usize,
        codes_differ: usize,
        vec_total: usize,
        vec_differ: usize,
        values_identical: usize,
        values: usize,
        max_abs: f32,
        max_rel_rms: f32,
        max_code_step: i32,
    }
    let mut aggs: std::collections::BTreeMap<(bool, u32), Agg> = Default::default();
    let mut at = 0usize;
    for (i, c) in cases.iter().enumerate() {
        let payload = T * T * c.bits as usize / 8;
        let part = &rec[at..at + payload + 3 * T * 2];
        at += payload + 3 * T * 2;
        let codes_ref = unpack(&part[..payload], T * T, c.bits as usize);
        let vec = |k: usize| -> [u16; T] {
            let mut out = [0u16; T];
            for j in 0..T {
                out[j] = u16::from_le_bytes([part[payload + k * 256 + j * 2], part[payload + k * 256 + j * 2 + 1]]);
            }
            out
        };
        let (scale_ref, zero_ref, other_ref) = if c.is_k { (vec(0), vec(1), vec(2)) } else { (vec(1), vec(2), vec(0)) };

        let ours = kvarn::quantise(&kvarn::normalise(&c.tile, 16), c.bits);
        let deq_ours = kvarn::dequantise(&ours);
        let deq_ref = &deq[i * T * T..(i + 1) * T * T];

        let a = aggs.entry((c.is_k, c.bits)).or_insert(Agg { tiles: 0, tiles_identical_codes: 0, codes: 0, codes_differ: 0, vec_total: 0, vec_differ: 0, values_identical: 0, values: 0, max_abs: 0.0, max_rel_rms: 0.0, max_code_step: 0 });
        a.tiles += 1;
        let differ = ours.codes.iter().zip(&codes_ref).filter(|(x, y)| x != y).count();
        a.tiles_identical_codes += usize::from(differ == 0);
        a.codes += T * T;
        a.codes_differ += differ;
        for (x, y) in ours.codes.iter().zip(&codes_ref) {
            a.max_code_step = a.max_code_step.max((i32::from(*x) - i32::from(*y)).abs());
        }
        for (mine, theirs) in [(&ours.scale, &scale_ref), (&ours.zero, &zero_ref), (&ours.other, &other_ref)] {
            a.vec_total += T;
            a.vec_differ += mine.iter().zip(theirs.iter()).filter(|(x, y)| x != y).count();
        }
        a.values += T * T;
        a.values_identical += deq_ours.iter().zip(deq_ref).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
        let rms = (deq_ref.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / (T * T) as f64).sqrt() as f32;
        let maxd = deq_ours.iter().zip(deq_ref).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        a.max_abs = a.max_abs.max(maxd);
        a.max_rel_rms = a.max_rel_rms.max(maxd / rms.max(1e-30));
    }
    println!("tiles (rows of the K tile are channels, of the V tile tokens), real data, layers {:?}:", [0, layers.len() / 2, layers.len() - 1].map(|l| layers[l].il));
    println!("{:>4} {:>4} {:>6} {:>12} {:>16} {:>14} {:>16} {:>12} {:>12}", "kind", "bits", "tiles", "codes equal", "tiles all-equal", "f16 vec equal", "values bit-equal", "max |diff|", "max / rms");
    for ((is_k, bits), a) in &aggs {
        println!(
            "{:>4} {:>4} {:>6} {:>11.4}% {:>10}/{:<5} {:>13.4}% {:>15.4}% {:>12.2e} {:>12.2e}",
            if *is_k { "K" } else { "V" },
            bits,
            a.tiles,
            100.0 * (1.0 - a.codes_differ as f64 / a.codes as f64),
            a.tiles_identical_codes,
            a.tiles,
            100.0 * (1.0 - a.vec_differ as f64 / a.vec_total as f64),
            100.0 * a.values_identical as f64 / a.values as f64,
            a.max_abs,
            a.max_rel_rms,
        );
        println!("       largest code difference: {}", a.max_code_step);
    }
}

/// The built codec (`encode_tile`/`decode_tile`) against the row codec and
/// against this tool's what-if of the same idea (mean of the tile in the
/// original domain, tile of 128).
fn built_codec(layers: &[Layer]) {
    println!("== built tile codec (encode_tile / decode_tile) against rows and against the what-if");
    let layer = &layers[0];
    let hd = layer.dims.hd;
    let rows_full = layer.n / 128 * 128;
    for (name, x) in [("K", layer.k_head(0)), ("V", layer.v_head(0))] {
        let q8 = codecs::built(&x, hd, 8);
        let rows8 = codecs::engine(&x, hd, 8);
        let same = q8.iter().zip(&rows8).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        println!("layer {} {name}: q8 built against rows: {} of {} values bit-identical", layer.il, same, q8.len());
        for bits in [6u32, 5, 4] {
            let built = codecs::built(&x, hd, bits);
            let rows = codecs::engine(&x, hd, bits);
            let what_if = codecs::engine_tile_mean(&x, hd, bits, 128, false);
            let db = |y: &[f32]| {
                let (s, e) = codecs::sums(&x, y, hd, rows_full);
                10.0 * (s / e).log10()
            };
            let (mut num, mut den) = (0.0f64, 0.0f64);
            let mut max = 0.0f32;
            let mut differ = 0usize;
            let rms = (x[..rows_full * hd].iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / (rows_full * hd) as f64).sqrt() as f32;
            for ((a, b), o) in built[..rows_full * hd].iter().zip(&what_if[..rows_full * hd]).zip(&x[..rows_full * hd]) {
                num += f64::from(a - b).powi(2);
                den += f64::from(a - o).powi(2);
                max = max.max((a - b).abs());
                differ += usize::from((a - b).abs() > 1e-4 * rms);
            }
            println!(
                "layer {} {name} q{bits}: SNR rows {:.3} dB, built {:.3} dB, what-if {:.3} dB; built against what-if: rms difference / rms error of built {:.3}, max abs {:.2e}, {:.2}% of the values differ by more than 1e-4 of the rms",
                layer.il,
                db(&rows),
                db(&built),
                db(&what_if),
                (num / den).sqrt(),
                max,
                100.0 * differ as f64 / (rows_full * hd) as f64
            );
        }
    }
}

pub fn run(args: &[String]) {
    let dir = PathBuf::from(arg(args, "--data").expect("--data"));
    let dims = dims_from(args);
    let layers: Vec<Layer> = data::list_layers(&dir).into_iter().map(|il| data::load_layer(&dir, il, dims)).collect();
    attention_vs_model(&layers);
    absmax_is_q8(&layers);
    built_codec(&layers);
    if let Some(r) = arg(args, "--kvarn-ref") {
        let work = PathBuf::from(arg(args, "--work").expect("--work"));
        kvarn_against_original(&layers, r, &work);
    }
}
