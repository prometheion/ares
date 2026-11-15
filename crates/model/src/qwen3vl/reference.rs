//! The CPU reference of the pass: pixels in, the rows the language model
//! reads out.
//!
//! Values are `f32`. Every sum (a matrix product, a norm's mean and variance,
//! the softmax and the weighted values of the attention) is accumulated in
//! `f64` and rounded to `f32` once; the tanh, the exponential and the rotation
//! are computed in `f64` too. The order of every sum is fixed by the length of
//! the data alone, so the rows do not depend on the number of threads.
//! Biases and the residual are added in `f32`, as separate steps after the
//! rounded product.
//!
//! Two things are done in `f32` exactly as llama.cpp does them, because they
//! are tables rather than sums: the position table's bilinear resize and the
//! angles of the rotary rotation (a chain of `f32` multiplications).
//!
//! [`Arithmetic::ForkCpu`] adds the roundings of llama.cpp's CPU path so that
//! its numbers can be compared closely; it is not what a device should match.

use std::ops::Range;

use ares_core::GgmlType;
use ares_quant::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};

use crate::qwen3vl::config::Config;
use crate::qwen3vl::error::{Error, Result};
use crate::qwen3vl::image::{Limits, normalize, resize_bicubic, target_size};
use crate::qwen3vl::weights::{Linear, Norm, Weights};

/// Which roundings the reference applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arithmetic {
    /// The pass as defined: no value is rounded below `f32`.
    Exact,
    /// As llama.cpp computes it on the CPU without flash attention: pixels
    /// and patch kernels rounded to f16, the input of every matrix product
    /// rounded to the type the matrix is stored in (bf16 for a BF16 matrix),
    /// and the GELU through f16 on both sides.
    ForkCpu,
}

/// A point of the pass whose values [`Reference::encode_traced`] hands out.
/// The values are rows of the tower's width in the merger's patch order,
/// unless said otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The patches after the patch bias.
    Patches,
    /// With the learned positions added: the input of the first block.
    Positions,
    /// The first norm of block `.0`.
    Ln1(usize),
    /// Query, key and value before the rotation: rows of three times the width.
    Qkv(usize),
    /// The rotated queries.
    QueryRotated(usize),
    /// The rotated keys.
    KeyRotated(usize),
    /// The attention's weighted values, before the output matrix.
    Attention(usize),
    /// The attention's output, bias included.
    AttentionOut(usize),
    /// The residual after the attention.
    Residual(usize),
    /// The second norm.
    Ln2(usize),
    /// The MLP's hidden layer before the GELU: rows of `n_ff`.
    Hidden(usize),
    /// The same after the GELU.
    Gelu(usize),
    /// The MLP's output, bias included.
    Mlp(usize),
    /// The residual after the block.
    Block(usize),
    /// After the last norm.
    PostNorm,
    /// The merger's hidden layer before the GELU: rows of four times the width.
    MergerHidden,
    /// The same after the GELU.
    MergerGelu,
    /// The output rows.
    Output,
}

impl Stage {
    /// The name llama.cpp gives the same tensor in its graph, where it has one.
    pub fn fork_name(self, n_layer: usize) -> Option<String> {
        let block = |name: &str, layer: usize| Some(format!("{name}-{layer}"));
        match self {
            Stage::Patches => Some("patch_bias".to_owned()),
            Stage::Positions => Some("inp_pos_emb".to_owned()),
            Stage::Ln1(l) => block("ln1", l),
            Stage::QueryRotated(l) => block("Qcur_rope", l),
            Stage::KeyRotated(l) => block("Kcur_rope", l),
            Stage::Attention(l) => block("kqv_out", l),
            Stage::AttentionOut(l) => block("attn_out", l),
            Stage::Residual(l) => block("ffn_inp", l),
            Stage::Ln2(l) => block("ffn_inp_normed", l),
            Stage::Hidden(l) => block("ffn_up_b", l),
            Stage::Gelu(l) => block("ffn_gelu", l),
            Stage::Mlp(l) => block("ffn_out", l),
            Stage::Block(l) => block("layer_out", l),
            Stage::PostNorm => block("norm_b", n_layer),
            Stage::MergerHidden => Some("ffn_up_b".to_owned()),
            Stage::MergerGelu => Some("ffn_gelu".to_owned()),
            Stage::Qkv(_) | Stage::Output => None,
        }
    }
}

/// The rows of one image: `nx * ny` rows of `width` values, row `i` at grid
/// cell `(i % nx, i / nx)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    /// Rows per line of the grid.
    pub nx: usize,
    /// Lines of the grid.
    pub ny: usize,
    /// Values per row: the language model's embedding width.
    pub width: usize,
    /// The values, row after row.
    pub data: Vec<f32>,
}

/// The reference of one projector.
#[derive(Clone, Copy, Debug)]
pub struct Reference<'a> {
    config: &'a Config,
    weights: &'a Weights<'a>,
    threads: usize,
    arithmetic: Arithmetic,
}

impl<'a> Reference<'a> {
    /// A reference in [`Arithmetic::Exact`] on all cores.
    pub fn new(config: &'a Config, weights: &'a Weights<'a>) -> Self {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        Self { config, weights, threads, arithmetic: Arithmetic::Exact }
    }

    /// The same on up to `threads` threads. The rows do not depend on it.
    pub fn with_threads(self, threads: usize) -> Self {
        Self { threads: threads.max(1), ..self }
    }

    /// The same with other roundings.
    pub fn with_arithmetic(self, arithmetic: Arithmetic) -> Self {
        Self { arithmetic, ..self }
    }

    /// The rows of an image given as 8-bit RGB pixels of any size: the size
    /// rule, the resize, the normalisation and the pass.
    pub fn encode_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        limits: Limits,
    ) -> Result<Rows> {
        if rgb.len() != width.saturating_mul(height).saturating_mul(3) {
            return Err(image(width, height, "does not have three bytes for every pixel"));
        }
        let (w, h) = target_size(self.config, limits, width, height)?;
        let resized = resize_bicubic(rgb, width, height, w, h);
        self.encode(&normalize(self.config, &resized), w, h)
    }

    /// The rows of normalised pixels (three `f32` a pixel, row after row)
    /// whose sides are multiples of [`Config::align`].
    pub fn encode(&self, pixels: &[f32], width: usize, height: usize) -> Result<Rows> {
        self.encode_traced(pixels, width, height, &mut |_, _| {})
    }

    /// [`Reference::encode`], handing every [`Stage`] to `tap` as it is reached.
    pub fn encode_traced(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
        tap: &mut dyn FnMut(Stage, &[f32]),
    ) -> Result<Rows> {
        let (c, w) = (self.config, self.weights);
        let align = c.align();
        if width == 0
            || height == 0
            || !width.is_multiple_of(align)
            || !height.is_multiple_of(align)
        {
            return Err(image(width, height, &format!("is not made of squares of {align} pixels")));
        }
        if pixels.len() != width * height * 3 {
            return Err(image(width, height, "does not have three values for every pixel"));
        }
        let (pw, ph) = (width / c.patch, height / c.patch);
        let (nx, ny) = (pw / 2, ph / 2);
        let (n_patches, e) = (pw * ph, c.n_embd);
        // Where patch `i` of the merger's order lies in the grid: (row, column).
        let order: Vec<(usize, usize)> = (0..n_patches)
            .map(|i| (2 * (i / 4 / nx) + i % 4 / 2, 2 * (i / 4 % nx) + i % 2))
            .collect();

        let mut x = self.patches(pixels, width, &order);
        tap(Stage::Patches, &x);
        add(&mut x, &self.positions(pw, ph, &order));
        tap(Stage::Positions, &x);

        let rotation = self.rotation(&order);
        for (l, block) in w.blocks.iter().enumerate() {
            let normed = layer_norm(&block.ln1, c.eps, &x);
            tap(Stage::Ln1(l), &normed);
            let qkv = self.linear(&block.qkv, &normed, n_patches);
            tap(Stage::Qkv(l), &qkv);
            let query = self.rotate(&qkv, 0, &rotation);
            tap(Stage::QueryRotated(l), &query);
            let key = self.rotate(&qkv, e, &rotation);
            tap(Stage::KeyRotated(l), &key);
            let attended = self.attention(&query, &key, &qkv);
            tap(Stage::Attention(l), &attended);
            let out = self.linear(&block.out, &attended, n_patches);
            tap(Stage::AttentionOut(l), &out);
            add(&mut x, &out);
            tap(Stage::Residual(l), &x);

            let normed = layer_norm(&block.ln2, c.eps, &x);
            tap(Stage::Ln2(l), &normed);
            let mut hidden = self.linear(&block.up, &normed, n_patches);
            tap(Stage::Hidden(l), &hidden);
            self.gelu(&mut hidden);
            tap(Stage::Gelu(l), &hidden);
            let mlp = self.linear(&block.down, &hidden, n_patches);
            tap(Stage::Mlp(l), &mlp);
            add(&mut x, &mlp);
            tap(Stage::Block(l), &x);
        }

        let x = layer_norm(&w.post_ln, c.eps, &x);
        tap(Stage::PostNorm, &x);
        // Four consecutive patches are one group: its row of `4 * n_embd`.
        let mut hidden = self.linear(&w.mm0, &x, nx * ny);
        tap(Stage::MergerHidden, &hidden);
        self.gelu(&mut hidden);
        tap(Stage::MergerGelu, &hidden);
        let data = self.linear(&w.mm2, &hidden, nx * ny);
        tap(Stage::Output, &data);
        Ok(Rows { nx, ny, width: c.n_out, data })
    }

    /// The patches in the merger's order, through the patch matrix, plus the
    /// patch bias. The matrix is the sum of the two kernels, formed in `f32`.
    fn patches(&self, pixels: &[f32], width: usize, order: &[(usize, usize)]) -> Vec<f32> {
        let (c, w) = (self.config, self.weights);
        let fork = self.arithmetic == Arithmetic::ForkCpu;
        let through = |v: f32| if fork { f16_to_f32(f32_to_f16(v)) } else { v };
        let matrix: Vec<f32> =
            w.patch[0].iter().zip(&w.patch[1]).map(|(a, b)| through(*a) + through(*b)).collect();
        let mut rows = Vec::with_capacity(order.len() * c.patch_len());
        for &(py, px) in order {
            for channel in 0..3 {
                for ky in 0..c.patch {
                    let start = ((py * c.patch + ky) * width + px * c.patch) * 3 + channel;
                    rows.extend((0..c.patch).map(|kx| through(pixels[start + kx * 3])));
                }
            }
        }
        let mut x = matmul(self.threads, &matrix, c.n_embd, c.patch_len(), &rows, order.len());
        add_bias(&mut x, &w.patch_bias);
        x
    }

    /// The learned position of every patch: the table itself when the grid
    /// has its size, else resized bilinearly with aligned corners, in `f32`
    /// and in ggml's order of operations.
    fn positions(&self, pw: usize, ph: usize, order: &[(usize, usize)]) -> Vec<f32> {
        let (e, side, table) = (self.config.n_embd, self.config.table_side, &self.weights.position);
        let at = |y: usize, x: usize| &table[(y * side + x) * e..(y * side + x + 1) * e];
        let mut out = Vec::with_capacity(order.len() * e);
        if pw == side && ph == side {
            order.iter().for_each(|&(py, px)| out.extend_from_slice(at(py, px)));
            return out;
        }
        let factor = |n: usize| {
            if n > 1 && side > 1 {
                (n - 1) as f32 / (side - 1) as f32
            } else {
                n as f32 / side as f32
            }
        };
        // The two table cells around a coordinate and the weight of the second.
        let around = |i: usize, sf: f32| {
            let pos = i as f32 / sf;
            let lo = (pos.floor() as i64).clamp(0, side as i64 - 1);
            let hi = (pos.floor() as i64 + 1).clamp(0, side as i64 - 1);
            (lo as usize, hi as usize, (pos - lo as f32).clamp(0.0, 1.0))
        };
        let (sfx, sfy) = (factor(pw), factor(ph));
        for &(py, px) in order {
            let ((y0, y1, dy), (x0, x1, dx)) = (around(py, sfy), around(px, sfx));
            let (a, b, c, d) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
            out.extend((0..e).map(|i| {
                a[i] * (1.0 - dx) * (1.0 - dy)
                    + b[i] * dx * (1.0 - dy)
                    + c[i] * (1.0 - dx) * dy
                    + d[i] * dx * dy
            }));
        }
        out
    }

    /// Cosine and sine of every rotated pair of every patch: a head of `d`
    /// values has `d / 2` pairs `(k, k + d / 2)`; the first half of the pairs
    /// turns by the patch's row, the second by its column, pair `j` of a half
    /// by `position * base^(-j / (d / 4))` with base 10000. The angle is
    /// built as llama.cpp builds it, by multiplying on in `f32`.
    fn rotation(&self, order: &[(usize, usize)]) -> Vec<(f32, f32)> {
        let pairs = self.config.head_dim() / 2;
        let step = 10000f32.powf(-2.0 / pairs as f32);
        let mut table = Vec::with_capacity(order.len() * pairs);
        for &(row, col) in order {
            let mut theta = row as f32;
            for k in 0..pairs {
                if k == pairs / 2 {
                    theta = col as f32;
                }
                table.push((theta.cos(), theta.sin()));
                theta *= step;
            }
        }
        table
    }

    /// The part of `qkv` that starts at `offset` in every row, each head rotated.
    fn rotate(&self, qkv: &[f32], offset: usize, rotation: &[(f32, f32)]) -> Vec<f32> {
        let (e, pairs) = (self.config.n_embd, self.config.head_dim() / 2);
        let mut out = Vec::with_capacity(qkv.len() / 3);
        for (row, angles) in qkv.chunks_exact(3 * e).zip(rotation.chunks_exact(pairs)) {
            for head in row[offset..offset + e].chunks_exact(2 * pairs) {
                let (lo, hi) = head.split_at(pairs);
                let turned = |sign: f64, a: &[f32], b: &[f32]| {
                    let values = a.iter().zip(b).zip(angles).map(move |((a, b), (cos, sin))| {
                        (f64::from(*a) * f64::from(*cos) + sign * f64::from(*b) * f64::from(*sin))
                            as f32
                    });
                    values.collect::<Vec<f32>>()
                };
                out.extend(turned(-1.0, lo, hi));
                out.extend(turned(1.0, hi, lo));
            }
        }
        out
    }

    /// Full softmax attention of every patch over every patch, head by head:
    /// `sum_j softmax_j(q . k_j / sqrt(d)) v_j`, with the values taken from
    /// the last third of `qkv`.
    fn attention(&self, query: &[f32], key: &[f32], qkv: &[f32]) -> Vec<f32> {
        let (e, d) = (self.config.n_embd, self.config.head_dim());
        let n = query.len() / e;
        let scale = f64::from(1.0 / (d as f32).sqrt());
        let mut out = vec![0.0f32; n * e];
        for head in 0..self.config.n_head {
            let of = |rows: &[f32], width: usize, offset: usize| -> Vec<f32> {
                let start = offset + head * d;
                rows.chunks_exact(width).flat_map(|r| &r[start..start + d]).copied().collect()
            };
            let (q, k, v) = (of(query, e, 0), of(key, e, 0), of(qkv, 3 * e, 2 * e));
            let attended = parallel(self.threads, n, |range| {
                fast(|| attend(&q[range.start * d..range.end * d], &k, &v, d, scale))
            });
            for (dst, src) in out.chunks_exact_mut(e).zip(attended.chunks_exact(d)) {
                dst[head * d..(head + 1) * d].copy_from_slice(src);
            }
        }
        out
    }

    /// `x` through a matrix, plus its bias: `n` rows.
    fn linear(&self, layer: &Linear<'_>, x: &[f32], n: usize) -> Vec<f32> {
        let m = &layer.weight;
        let rounded: Vec<f32>;
        let x = match (self.arithmetic, m.ty()) {
            (Arithmetic::ForkCpu, GgmlType::BF16) => {
                rounded = x.iter().map(|v| bf16_to_f32(f32_to_bf16(*v))).collect();
                &rounded
            }
            (Arithmetic::ForkCpu, GgmlType::F16) => {
                rounded = x.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
                &rounded
            }
            _ => x,
        };
        let mut y = matmul(self.threads, &m.decode(), m.rows(), m.cols(), x, n);
        add_bias(&mut y, &layer.bias);
        y
    }

    /// The tanh GELU of every value, in place.
    fn gelu(&self, x: &mut [f32]) {
        let fork = self.arithmetic == Arithmetic::ForkCpu;
        let per = x.len().div_ceil(self.threads).max(1);
        std::thread::scope(|scope| {
            for part in x.chunks_mut(per) {
                scope.spawn(move || {
                    for v in part {
                        *v = if fork { gelu_fork(*v) } else { gelu(*v) };
                    }
                });
            }
        });
    }
}

fn image(width: usize, height: usize, problem: &str) -> Error {
    Error::Image { width, height, problem: problem.to_owned() }
}

/// `0.5 x (1 + tanh(sqrt(2 / pi) x (1 + 0.044715 x^2)))`.
pub fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    let inner = (2.0 / std::f64::consts::PI).sqrt() * x * (1.0 + 0.044715 * x * x);
    (0.5 * x * (1.0 + inner.tanh())) as f32
}

/// The GELU as llama.cpp's CPU path computes it: nothing below -10, the
/// identity above 10, and between them a table entry, which is the `f32`
/// formula of the input rounded to f16, itself rounded to f16.
pub fn gelu_fork(x: f32) -> f32 {
    if x <= -10.0 {
        return 0.0;
    }
    if x >= 10.0 {
        return x;
    }
    let x = f16_to_f32(f32_to_f16(x));
    let inner = 0.797_884_6_f32 * x * (1.0 + 0.044715 * x * x);
    f16_to_f32(f32_to_f16(0.5 * x * (1.0 + inner.tanh())))
}

/// Mean and variance over each row, then weight and bias:
/// `(x - mean) / sqrt(variance + eps) * weight + bias`.
fn layer_norm(norm: &Norm, eps: f32, x: &[f32]) -> Vec<f32> {
    let n = norm.weight.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(n) {
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
        let variance = row.iter().map(|v| (f64::from(*v) - mean).powi(2)).sum::<f64>() / n as f64;
        let scale = 1.0 / (variance + f64::from(eps)).sqrt();
        out.extend(row.iter().zip(&norm.weight).zip(&norm.bias).map(|((v, w), b)| {
            ((f64::from(*v) - mean) * scale * f64::from(*w) + f64::from(*b)) as f32
        }));
    }
    out
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

fn add_bias(x: &mut [f32], bias: &[f32]) {
    x.chunks_exact_mut(bias.len()).for_each(|row| add(row, bias));
}

/// The dot product of two `f32` vectors in `f64`: element `i` of the whole
/// multiples of sixteen goes to accumulator `i % 16`, the accumulators are
/// added in a fixed tree, and the remaining elements follow in order. Every
/// product is exact in `f64`.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = [0.0f64; 16];
    let ((a16, a_tail), (b16, b_tail)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    for (x, y) in a16.iter().zip(b16) {
        for lane in 0..16 {
            acc[lane] += f64::from(x[lane]) * f64::from(y[lane]);
        }
    }
    let mut sum = 0.0;
    for quad in acc.as_chunks::<4>().0 {
        sum += (quad[0] + quad[1]) + (quad[2] + quad[3]);
    }
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum += f64::from(*x) * f64::from(*y);
    }
    sum
}

/// `y[t] = W x[t]` for `n` rows of `cols` values: every output is one [`dot`]
/// rounded to `f32`. The loops are cut into tiles that stay in a core's
/// cache; a tile changes which products are computed next, not their sums.
fn matmul(threads: usize, w: &[f32], rows: usize, cols: usize, x: &[f32], n: usize) -> Vec<f32> {
    assert_eq!((w.len(), x.len()), (rows * cols, n * cols), "matrix or rows of the wrong size");
    let tile = (65_536 / cols).max(1);
    parallel(threads, n, |range| {
        let mut y = vec![0.0f32; range.len() * rows];
        let x = &x[range.start * cols..range.end * cols];
        fast(|| {
            for (xs, ys) in x.chunks(tile * cols).zip(y.chunks_mut(tile * rows)) {
                for (r0, ws) in w.chunks(tile * cols).enumerate() {
                    for (xr, yr) in xs.chunks_exact(cols).zip(ys.chunks_exact_mut(rows)) {
                        for (r, wr) in ws.chunks_exact(cols).enumerate() {
                            yr[r0 * tile + r] = dot(wr, xr) as f32;
                        }
                    }
                }
            }
        });
        y
    })
}

/// The attention of the queries `q` (rows of `d`) of one head over all keys
/// and values of that head. Per query: the scores `q . k_j * scale`, their
/// largest value taken off, the exponentials summed in the order of `j`, the
/// values weighted and summed in the order of `j`, and the division.
#[inline(always)]
fn attend(q: &[f32], k: &[f32], v: &[f32], d: usize, scale: f64) -> Vec<f32> {
    const QUERIES: usize = 16;
    const KEYS: usize = 256;
    let n = k.len() / d;
    let mut out = Vec::with_capacity(q.len());
    let mut scores = vec![0.0f64; QUERIES * n];
    let mut sums = vec![0.0f64; QUERIES * d];
    for qs in q.chunks(QUERIES * d) {
        for (block, ks) in k.chunks(KEYS * d).enumerate() {
            for (qr, row) in qs.chunks_exact(d).zip(scores.chunks_exact_mut(n)) {
                for (kr, s) in ks.chunks_exact(d).zip(&mut row[block * KEYS..]) {
                    *s = dot(qr, kr) * scale;
                }
            }
        }
        let count = qs.len() / d;
        let mut totals = [0.0f64; QUERIES];
        for (row, total) in scores.chunks_exact_mut(n).zip(&mut totals).take(count) {
            let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for s in row {
                *s = (*s - top).exp();
                *total += *s;
            }
        }
        sums.fill(0.0);
        for (block, vs) in v.chunks(KEYS * d).enumerate() {
            for (row, sum) in scores.chunks_exact(n).zip(sums.chunks_exact_mut(d)).take(count) {
                for (vr, p) in vs.chunks_exact(d).zip(&row[block * KEYS..]) {
                    for (s, value) in sum.iter_mut().zip(vr) {
                        *s += p * f64::from(*value);
                    }
                }
            }
        }
        for (sum, total) in sums.chunks_exact(d).zip(&totals).take(count) {
            out.extend(sum.iter().map(|s| (s / total) as f32));
        }
    }
    out
}

/// Runs `work` on contiguous ranges of `0..n` on up to `threads` threads and
/// concatenates what the ranges return, in order.
fn parallel(threads: usize, n: usize, work: impl Fn(Range<usize>) -> Vec<f32> + Sync) -> Vec<f32> {
    let threads = threads.min(n).max(1);
    if threads == 1 {
        return work(0..n);
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = (0..threads)
            .map(|t| scope.spawn(move || work((t * per).min(n)..((t + 1) * per).min(n))))
            .collect();
        let join = |h: std::thread::ScopedJoinHandle<'_, Vec<f32>>| match h.join() {
            Ok(part) => part,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        handles.into_iter().flat_map(join).collect()
    })
}

/// Runs `work` compiled for AVX2 where the CPU has it. Rust never contracts a
/// multiplication and an addition into one operation and keeps the order of
/// every sum, so the result has the same bits either way; only the time
/// differs.
fn fast<R>(work: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        fn wide<R>(work: impl FnOnce() -> R) -> R {
            work()
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU was just found to have AVX2, the one feature
            // the callee is compiled for.
            return unsafe { wide(work) };
        }
    }
    work()
}
