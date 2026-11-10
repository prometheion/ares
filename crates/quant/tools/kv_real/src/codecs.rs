//! The codecs under test. Every function takes rows `[n][hd]` in the original
//! domain and returns what attention reads back, rows `[n][hd]` in the
//! original domain (decoded and, where the format rotates, un-rotated: a
//! rotation does not change a dot product, so this equals the rotated-domain
//! evaluation of the kernels up to float rounding).

use ares_quant::{
    KV_TILE_CELLS, KvFormat, decode_row, decode_row_rotated, decode_tile, encode_row, encode_tile, f16_to_f32,
    f32_to_f16, rotate_groups,
};

use crate::kvarn::{self, T};

fn f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// Exact cells of the engine (and of the absolute-maximum baseline): `f16`.
pub fn exact_engine(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| f16(*v)).collect()
}

/// Exact cells of kvarn: the rotated head is stored as `f16` (the stage).
pub fn exact_kvarn(x: &[f32], hd: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(hd) {
        let mut r = row.to_vec();
        kvarn::rotate_head(&mut r);
        r.iter_mut().for_each(|v| *v = f16(*v));
        kvarn::rotate_head(&mut r);
        out.extend_from_slice(&r);
    }
    out
}

/// The engine's format as shipped (`ares_quant::encode_row`/`decode_row`).
pub fn engine(x: &[f32], hd: usize, bits: u32) -> Vec<f32> {
    let format = KvFormat::quantized(bits).expect("bits");
    let mut bytes = vec![0u8; format.row_bytes(hd).unwrap()];
    let mut out = vec![0.0f32; x.len()];
    for (row, dst) in x.chunks_exact(hd).zip(out.chunks_exact_mut(hd)) {
        encode_row(format, row, &mut bytes).unwrap();
        decode_row(format, &bytes, dst).unwrap();
    }
    out
}

/// The smallest positive f16 not below `q` (the scale rule of the specification).
fn scale_up(q: f32) -> f32 {
    let nearest = f32_to_f16(q);
    if nearest >= 0x7c00 {
        return f16_to_f32(0x7bff);
    }
    let up = if nearest == 0 || f16_to_f32(nearest) < q { nearest + 1 } else { nearest };
    f16_to_f32(up.min(0x7bff))
}

/// The engine's format with the old absolute-maximum scale (`BASE` of
/// `tests/kv_techniques.rs`): the group rotation, then a scale from the largest
/// magnitude rounded up to f16 and the symmetric code range. At 8 bits this is
/// `q8` of the library, bit for bit (checked by `validate`). With `rotate`
/// false the group rotation is left out.
pub fn absmax(x: &[f32], hd: usize, bits: u32, rotate: bool) -> Vec<f32> {
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    let mut out = Vec::with_capacity(x.len());
    let mut r = vec![0.0f32; hd];
    for row in x.chunks_exact(hd) {
        r.copy_from_slice(row);
        if rotate {
            rotate_groups(&mut r).unwrap();
        }
        for g in r.chunks_exact_mut(32) {
            let amax = g.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            let s = scale_up(amax / qmax);
            for v in g.iter_mut() {
                *v = (*v / s).round_ties_even().clamp(-qmax, qmax) * s;
            }
        }
        if rotate {
            rotate_groups(&mut r).unwrap();
        }
        out.extend_from_slice(&r);
    }
    out
}

/// kvarn's tiles of one K or V sequence, normalised once and quantised for
/// every width. Only whole groups of 128 rows are tiles; the rows after the
/// last whole group come back as `f16` of the rotated head (kvarn keeps them
/// in its stage).
pub struct KvarnTiles {
    hd: usize,
    n: usize,
    is_k: bool,
    /// The rotated head rows rounded to f16 (what the stage holds), `[n][hd]`.
    rot: Vec<f32>,
    /// `[group][slice]`.
    norm: Vec<kvarn::Normalised>,
}

impl KvarnTiles {
    pub fn new(x: &[f32], hd: usize, is_k: bool) -> Self {
        let n = x.len() / hd;
        let mut rot = Vec::with_capacity(x.len());
        for row in x.chunks_exact(hd) {
            let mut r = row.to_vec();
            kvarn::rotate_head(&mut r);
            r.iter_mut().for_each(|v| *v = f16(*v));
            rot.extend_from_slice(&r);
        }
        let slices = hd / T;
        let groups = n / T;
        let mut norm = Vec::new();
        for g in 0..groups {
            for s in 0..slices {
                let tile = Self::tile_of(&rot, hd, is_k, g, s);
                norm.push(kvarn::normalise(&tile, 16));
            }
        }
        KvarnTiles { hd, n, is_k, rot, norm }
    }

    /// K: `[channel][token]`; V: `[token][channel]`.
    pub fn tile_of(rot: &[f32], hd: usize, is_k: bool, g: usize, s: usize) -> Vec<f32> {
        let mut tile = vec![0.0f32; T * T];
        for tok in 0..T {
            let row = &rot[(g * T + tok) * hd + s * T..(g * T + tok) * hd + (s + 1) * T];
            for d in 0..T {
                if is_k {
                    tile[d * T + tok] = row[d];
                } else {
                    tile[tok * T + d] = row[d];
                }
            }
        }
        tile
    }

    pub fn decode(&self, bits: u32) -> Vec<f32> {
        let slices = self.hd / T;
        let groups = self.n / T;
        let mut rec = self.rot.clone();
        for g in 0..groups {
            for s in 0..slices {
                let deq = kvarn::dequantise(&kvarn::quantise(&self.norm[g * slices + s], bits));
                for tok in 0..T {
                    let dst = &mut rec[(g * T + tok) * self.hd + s * T..(g * T + tok) * self.hd + (s + 1) * T];
                    for d in 0..T {
                        dst[d] = if self.is_k { deq[d * T + tok] } else { deq[tok * T + d] };
                    }
                }
            }
        }
        for row in rec.chunks_exact_mut(self.hd) {
            kvarn::rotate_head(row);
        }
        rec
    }
}

/// Static per-channel statistics from the first `n0` rows: the mean and the
/// scale (rms of the centred values relative to their geometric mean,
/// clamped to 1/16 .. 16, to the power `alpha`).
pub struct StaticStats {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
}

pub fn static_stats(x: &[f32], hd: usize, n0: usize, alpha: f32, mean: bool, scale: bool) -> StaticStats {
    let mut mu = vec![0.0f32; hd];
    if mean {
        for row in x.chunks_exact(hd).take(n0) {
            for (m, v) in mu.iter_mut().zip(row) {
                *m += v;
            }
        }
        mu.iter_mut().for_each(|m| *m /= n0 as f32);
    }
    let mut s = vec![1.0f32; hd];
    if scale {
        let mut rms = vec![0.0f64; hd];
        for row in x.chunks_exact(hd).take(n0) {
            for c in 0..hd {
                rms[c] += f64::from(row[c] - mu[c]).powi(2);
            }
        }
        let rms: Vec<f64> = rms.iter().map(|v| (v / n0 as f64).sqrt().max(1e-12)).collect();
        let gm = (rms.iter().map(|v| v.ln()).sum::<f64>() / hd as f64).exp();
        for c in 0..hd {
            s[c] = ((rms[c] / gm).clamp(1.0 / 16.0, 16.0) as f32).powf(alpha);
        }
    }
    StaticStats { mean: mu, scale: s }
}

/// The engine's format on `(x - mean) / scale` with a static mean and scale
/// per channel from the first `n0` rows, for the rows from `n0` on (earlier
/// rows are stored plain, as they came before the statistics existed).
/// Reading: the query is multiplied by the scale once (K), or the output is
/// (V), and the mean cancels in the softmax (K) or is added once (V).
pub fn engine_static(x: &[f32], hd: usize, bits: u32, st: &StaticStats, n0: usize) -> Vec<f32> {
    let mut plain = engine(&x[..n0 * hd], hd, bits);
    let mut y = Vec::with_capacity(x.len() - n0 * hd);
    for row in x[n0 * hd..].chunks_exact(hd) {
        for c in 0..hd {
            y.push((row[c] - st.mean[c]) / st.scale[c]);
        }
    }
    let dec = engine(&y, hd, bits);
    for row in dec.chunks_exact(hd) {
        for c in 0..hd {
            plain.push(row[c] * st.scale[c] + st.mean[c]);
        }
    }
    plain
}

/// The two-axis normalisation scales of the tiles of a sequence, as stored
/// (f16), for `engine_tile_norm`. Tiles are 128 rows by 128 channels (the
/// head in slices of 128), `[group][slice]`.
pub struct TileScales {
    pub rotated: bool,
    /// The values the tiles were cut from: the original rows, or after the
    /// group rotation.
    z: Vec<f32>,
    rs: Vec<[f32; T]>,
    cs: Vec<[f32; T]>,
}

/// Normalises every whole tile of `x` in both directions as kvarn does it
/// (16 Sinkhorn iterations). With `rotated` the tiles are cut from the values
/// after the group rotation (kvarn's order: rotate, then normalise); otherwise
/// from the original values (before it).
pub fn tile_scales(x: &[f32], hd: usize, rotated: bool) -> TileScales {
    let n = x.len() / hd;
    let slices = hd / T;
    let z: Vec<f32> = if rotated {
        let mut z = x.to_vec();
        rotate_groups(&mut z).unwrap();
        z
    } else {
        x.to_vec()
    };
    let (mut rs, mut cs) = (Vec::new(), Vec::new());
    for g in 0..n / T {
        for s in 0..slices {
            let tile = KvarnTiles::tile_of(&z, hd, false, g, s); // [token][channel]
            let nm = kvarn::normalise(&tile, 16);
            rs.push(nm.s_row.map(f16));
            cs.push(nm.s_col.map(f16));
        }
    }
    TileScales { rotated, z, rs, cs }
}

/// The engine's format on tile-normalised rows: `value / (row scale * channel
/// scale)` goes through the codec and the scales are multiplied back. Cost of
/// the scales: 128 + 128 f16 per tile and slice, 0.25 bits per value. Rows
/// after the last whole group stay `f16`.
pub fn engine_tile_norm(x: &[f32], hd: usize, bits: u32, sc: &TileScales) -> Vec<f32> {
    let n = x.len() / hd;
    let slices = hd / T;
    let rotated = sc.rotated;
    let format = KvFormat::quantized(bits).unwrap();
    let mut out = exact_engine(x);
    let mut bytes = vec![0u8; format.row_bytes(hd).unwrap()];
    let mut dec = vec![0.0f32; hd];
    let mut u = vec![0.0f32; hd];
    for g in 0..n / T {
        for tok in 0..T {
            let at = (g * T + tok) * hd;
            for s in 0..slices {
                let (r, c) = (sc.rs[g * slices + s][tok], &sc.cs[g * slices + s]);
                for d in 0..T {
                    u[s * T + d] = sc.z[at + s * T + d] / (r * c[d]);
                }
            }
            if rotated {
                // the codec rotates its input: hand it the un-rotated values so
                // that it quantises the normalised rotated ones
                rotate_groups(&mut u).unwrap();
                encode_row(format, &u, &mut bytes).unwrap();
                decode_row_rotated(format, &bytes, &mut dec).unwrap();
            } else {
                encode_row(format, &u, &mut bytes).unwrap();
                decode_row(format, &bytes, &mut dec).unwrap();
            }
            for s in 0..slices {
                let (r, c) = (sc.rs[g * slices + s][tok], &sc.cs[g * slices + s]);
                for d in 0..T {
                    dec[s * T + d] *= r * c[d];
                }
            }
            if rotated {
                rotate_groups(&mut dec).unwrap();
            }
            out[at..at + hd].copy_from_slice(&dec);
        }
    }
    out
}

/// The engine's format on the rows minus the mean of their tile of `tile`
/// rows (one f16 vector per tile and head; rows after the last whole tile stay
/// `f16`). With `lagged` the offset of a tile is the mean of the tile before it
/// (so a row can be written when it arrives) and the first tile has none.
pub fn engine_tile_mean(x: &[f32], hd: usize, bits: u32, tile: usize, lagged: bool) -> Vec<f32> {
    let n = x.len() / hd;
    let mut out = exact_engine(x);
    let mean_of = |g: usize| -> Vec<f32> {
        let mut mean = vec![0.0f32; hd];
        for row in x[g * tile * hd..(g + 1) * tile * hd].chunks_exact(hd) {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += v;
            }
        }
        mean.iter_mut().for_each(|m| *m = f16(*m / tile as f32));
        mean
    };
    for g in 0..n / tile {
        let mean = if lagged {
            if g == 0 { vec![0.0f32; hd] } else { mean_of(g - 1) }
        } else {
            mean_of(g)
        };
        let rows = &x[g * tile * hd..(g + 1) * tile * hd];
        let y: Vec<f32> = rows.chunks_exact(hd).flat_map(|r| r.iter().zip(&mean).map(|(v, m)| v - m)).collect();
        let dec = engine(&y, hd, bits);
        for (dst, (d, m)) in out[g * tile * hd..(g + 1) * tile * hd].iter_mut().zip(dec.iter().zip(mean.iter().cycle())) {
            *dst = d + m;
        }
    }
    out
}

/// The engine's formats as built: `encode_tile`/`decode_tile` of `ares-quant`
/// on tiles of `KV_TILE_CELLS` (128) cells aligned to the start of the sequence,
/// the mean of the rotated rows stored as f16 for `q7` to `q4` (none for `q8`,
/// whose rows are those of `encode_row`). Rows after the last whole tile come
/// back as `f16` for the formats with a mean (a tile is stored when it is full)
/// and as rows for `q8`; the exact cells of a policy are chosen by the caller.
pub fn built(x: &[f32], hd: usize, bits: u32) -> Vec<f32> {
    let format = KvFormat::quantized(bits).unwrap();
    let n = x.len() / hd;
    let mut out = exact_engine(x);
    let mut mean = vec![0u8; format.mean_bytes(hd).unwrap()];
    let mut bytes = vec![0u8; KV_TILE_CELLS * format.row_bytes(hd).unwrap()];
    for g in 0..n / KV_TILE_CELLS {
        let at = g * KV_TILE_CELLS * hd;
        let rows = &x[at..at + KV_TILE_CELLS * hd];
        encode_tile(format, hd, rows, &mut mean, &mut bytes).unwrap();
        decode_tile(format, hd, &mean, &bytes, &mut out[at..at + KV_TILE_CELLS * hd]).unwrap();
    }
    if !format.has_mean() {
        // no mean, no tile: a format without one stores every cell as soon as it
        // is written (rows after the last whole tile included)
        let from = n / KV_TILE_CELLS * KV_TILE_CELLS * hd;
        let tail = engine(&x[from..], hd, bits);
        out[from..].copy_from_slice(&tail);
    }
    out
}

/// The engine's format with the midrange of each rotated group as an offset
/// (an asymmetric grid; one f16 per group, read cost: one term per group).
/// Subtracting a constant `z` from the 32 rotated values of a group equals
/// subtracting `z * sqrt(32)` from the first value of the group before the
/// rotation, which is how it is handed to the shipped codec.
pub fn engine_group_offset(x: &[f32], hd: usize, bits: u32) -> Vec<f32> {
    let format = KvFormat::quantized(bits).unwrap();
    let mut bytes = vec![0u8; format.row_bytes(hd).unwrap()];
    let mut out = Vec::with_capacity(x.len());
    let mut rot = vec![0.0f32; hd];
    let mut shifted = vec![0.0f32; hd];
    let mut dec = vec![0.0f32; hd];
    let spike = 32.0f32.sqrt();
    for row in x.chunks_exact(hd) {
        rot.copy_from_slice(row);
        rotate_groups(&mut rot).unwrap();
        shifted.copy_from_slice(row);
        let mut offsets = Vec::with_capacity(hd / 32);
        for (g, r) in rot.chunks_exact(32).enumerate() {
            let hi = r.iter().copied().fold(f32::MIN, f32::max);
            let lo = r.iter().copied().fold(f32::MAX, f32::min);
            let z = f16((hi + lo) / 2.0);
            offsets.push(z);
            shifted[g * 32] -= z * spike;
        }
        encode_row(format, &shifted, &mut bytes).unwrap();
        decode_row(format, &bytes, &mut dec).unwrap();
        for (g, z) in offsets.iter().enumerate() {
            dec[g * 32] += z * spike;
        }
        out.extend_from_slice(&dec);
    }
    out
}

/// 10 log10 of the signal power over the error power of rows `0..rows`.
pub fn sums(x: &[f32], y: &[f32], hd: usize, rows: usize) -> (f64, f64) {
    let (mut s, mut e) = (0.0f64, 0.0f64);
    for (a, b) in x[..rows * hd].iter().zip(&y[..rows * hd]) {
        s += f64::from(*a).powi(2);
        e += f64::from(*a - *b).powi(2);
    }
    (s, e)
}
