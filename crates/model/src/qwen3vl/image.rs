//! The pixel work in front of the tower: which size an image is resized to,
//! the resize, and the normalisation. Pixels are 8-bit RGB, row after row,
//! three bytes a pixel.
//!
//! All three follow the preprocessing of llama.cpp's `mtmd` for this projector
//! operation by operation, so that the same pixels reach the tower: the size
//! rule in `f32` as written there (`calc_size_preserved_ratio`), the resize in
//! the fixed-point arithmetic of Pillow's resampling that it transcribes
//! (`resize_pillow`), and `(value / 255 - mean) / std`.

use crate::qwen3vl::config::Config;
use crate::qwen3vl::error::{Error, Result};

/// How many rows an image may become.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// An image that would give fewer rows is enlarged.
    pub min_rows: usize,
    /// An image that would give more rows is shrunk.
    pub max_rows: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { min_rows: 8, max_rows: 4096 }
    }
}

/// The size, in pixels, that an image of `width` x `height` is resized to.
///
/// Both sides become multiples of [`Config::align`], rounded to the nearest
/// one (a tie goes to the even multiple) and at least one. A result with more
/// pixels than `max_rows` rows cover is replaced by the image scaled down by
/// the square root of the excess, each side rounded down; one with fewer than
/// `min_rows` rows by the image scaled up, each side rounded up. The aspect
/// ratio is kept only as far as the rounding allows. An image becomes
/// `(width / align) * (height / align)` rows.
pub fn target_size(
    config: &Config,
    limits: Limits,
    width: usize,
    height: usize,
) -> Result<(usize, usize)> {
    let image = |problem: &str| Error::Image { width, height, problem: problem.to_owned() };
    if width == 0 || height == 0 {
        return Err(image("has no pixels"));
    }
    let (Ok(w), Ok(h), Ok(f)) =
        (i32::try_from(width), i32::try_from(height), i32::try_from(config.align()))
    else {
        return Err(image("is larger than the size rule can count"));
    };
    let unit = f as f32;
    let round = |x: f32| i64::from((x / unit).round_ties_even() as i32) * i64::from(f);
    let ceil = |x: f32| i64::from((x / unit).ceil() as i32) * i64::from(f);
    let floor = |x: f32| i64::from((x / unit).floor() as i32) * i64::from(f);
    let area = i64::from(f) * i64::from(f);
    let (min_pixels, max_pixels) = (limits.min_rows as i64 * area, limits.max_rows as i64 * area);

    let mut w_bar = round(w as f32).max(i64::from(f));
    let mut h_bar = round(h as f32).max(i64::from(f));
    if h_bar * w_bar > max_pixels {
        let beta = (h as f32 * w as f32 / max_pixels as f32).sqrt();
        h_bar = floor(h as f32 / beta).max(i64::from(f));
        w_bar = floor(w as f32 / beta).max(i64::from(f));
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (h as f32 * w as f32)).sqrt();
        h_bar = ceil(h as f32 * beta);
        w_bar = ceil(w as f32 * beta);
    }
    Ok((w_bar as usize, h_bar as usize))
}

/// The fractional bits of a filter weight: 32 less the 8 of a pixel and 2 of
/// headroom for the sum.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// The bicubic kernel with `a = -0.5` (Pillow's; not the `-0.75` of PyTorch).
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// The filter of one direction: for each output pixel the first input pixel,
/// how many inputs it reads, and their fixed-point weights.
struct Filter {
    ksize: usize,
    /// `(first, count)` per output pixel.
    bounds: Vec<(usize, usize)>,
    /// `ksize` weights per output pixel, the first `count` of them used.
    weights: Vec<i32>,
}

impl Filter {
    fn new(in_size: usize, out_size: usize) -> Self {
        let scale = in_size as f64 / out_size as f64;
        // Shrinking widens the filter, enlarging keeps it.
        let filterscale = scale.max(1.0);
        let support = 2.0 * filterscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let fixed = f64::from(1u32 << PRECISION_BITS);
        let mut bounds = Vec::with_capacity(out_size);
        let mut weights = vec![0i32; out_size * ksize];
        let mut pre = vec![0.0f64; ksize];
        for xx in 0..out_size {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64);
            let count = (xmax - xmin) as usize;
            let mut total = 0.0;
            for (x, w) in pre[..count].iter_mut().enumerate() {
                *w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
                total += *w;
            }
            for (w, out) in pre[..count].iter().zip(&mut weights[xx * ksize..]) {
                let w = if total != 0.0 { w / total } else { *w };
                // Half added away from zero, then cut towards zero.
                *out = (w * fixed + if w < 0.0 { -0.5 } else { 0.5 }) as i32;
            }
            bounds.push((xmin as usize, count));
        }
        Self { ksize, bounds, weights }
    }

    fn taps(&self, out: usize) -> (usize, &[i32]) {
        let (first, count) = self.bounds[out];
        (first, &self.weights[out * self.ksize..out * self.ksize + count])
    }
}

fn clip8(sum: i32) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resizes 8-bit RGB pixels to `out_width` x `out_height` with the bicubic
/// resampling of Pillow: columns first, then rows, each a filter in 22-bit
/// fixed point whose result is rounded and cut to `0..=255` before the next
/// pass. A direction that keeps its size is not filtered.
///
/// # Panics
///
/// Panics when `rgb` is not `width * height` pixels or a size is zero.
pub fn resize_bicubic(
    rgb: &[u8],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<u8> {
    assert!(width > 0 && height > 0 && out_width > 0 && out_height > 0, "a size is zero");
    assert_eq!(rgb.len(), width * height * 3, "rgb must be {width} x {height} pixels");
    let half = 1i32 << (PRECISION_BITS - 1);
    let mut wide = rgb.to_vec();
    if out_width != width {
        let filter = Filter::new(width, out_width);
        wide = vec![0u8; out_width * height * 3];
        for (src, dst) in rgb.chunks_exact(width * 3).zip(wide.chunks_exact_mut(out_width * 3)) {
            for (xx, pixel) in dst.chunks_exact_mut(3).enumerate() {
                let (first, taps) = filter.taps(xx);
                let mut sums = [half; 3];
                for (p, k) in src[first * 3..].chunks_exact(3).zip(taps) {
                    for c in 0..3 {
                        sums[c] += i32::from(p[c]) * k;
                    }
                }
                for c in 0..3 {
                    pixel[c] = clip8(sums[c]);
                }
            }
        }
    }
    if out_height == height {
        return wide;
    }
    let filter = Filter::new(height, out_height);
    let row = out_width * 3;
    let mut out = vec![0u8; row * out_height];
    let mut acc = vec![0i32; row];
    for (yy, dst) in out.chunks_exact_mut(row).enumerate() {
        let (first, taps) = filter.taps(yy);
        acc.fill(half);
        for (src, k) in wide[first * row..].chunks_exact(row).zip(taps) {
            for (a, p) in acc.iter_mut().zip(src) {
                *a += i32::from(*p) * k;
            }
        }
        for (d, a) in dst.iter_mut().zip(&acc) {
            *d = clip8(*a);
        }
    }
    out
}

/// 8-bit RGB pixels as the tower's input: `(value / 255 - mean) / std` per
/// channel, in `f32` and in this order of operations.
pub fn normalize(config: &Config, rgb: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rgb.len());
    for pixel in rgb.chunks_exact(3) {
        for c in 0..3 {
            out.push((f32::from(pixel[c]) / 255.0 - config.mean[c]) / config.std[c]);
        }
    }
    out
}
