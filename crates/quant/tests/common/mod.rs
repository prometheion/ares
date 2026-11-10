//! Deterministic test data shared by the KV test files (`kv.rs` and
//! `kv_techniques.rs`): a generator and the shapes of rows built from it.
//!
//! Only integer operations and plain `f32` arithmetic are used (no `libm`
//! function; `sqrt` is correctly rounded by IEEE 754), so the numbers are the
//! same on every platform, which the golden fixture needs.

// Each test file uses some of these, not all.
#![allow(dead_code)]

use ares_quant::KV_GROUP_VALUES;

const GROUP: usize = KV_GROUP_VALUES;

/// A splitmix64 generator.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1), a multiple of 2^-24.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// Roughly standard normal: the sum of twelve uniforms minus six. Its tails
    /// stop at 6 sigma, which is why the data below adds outliers by hand.
    pub fn gaussian(&mut self) -> f32 {
        let mut sum = 0.0;
        for _ in 0..12 {
            sum += self.uniform();
        }
        sum - 6.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn sign(&mut self) -> f32 {
        if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

/// A row like a key or a value: roughly Gaussian, with a few outliers of 8 to
/// 16 times the typical magnitude (about one per 64 values) at random positions.
pub fn realistic_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mut row: Vec<f32> = (0..n).map(|_| 0.8 * rng.gaussian()).collect();
    for _ in 0..n / 64 {
        let at = rng.below(n);
        row[at] = rng.sign() * (8.0 + 8.0 * rng.uniform()) * 0.8;
    }
    row
}

/// One large value and the rest small: the case in which a rotation spreads the
/// big value over the group and drowns the small ones.
pub fn spike_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mut row: Vec<f32> = (0..n).map(|_| 0.05 * rng.gaussian()).collect();
    let at = rng.below(n);
    row[at] = rng.sign() * 100.0;
    row
}

/// Student's t with 3 degrees of freedom, scaled: a Gaussian divided by the
/// root of a mean of three squared Gaussians. Its variance is finite but its
/// fourth moment is not, so every row has a few values far beyond the rest.
pub fn heavy_tailed_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let g = rng.gaussian();
            let (a, b, c) = (rng.gaussian(), rng.gaussian(), rng.gaussian());
            0.5 * g / ((a * a + b * b + c * c) / 3.0).sqrt().max(0.05)
        })
        .collect()
}

/// A row with a non-zero mean: one to three times the typical magnitude, with a
/// random sign, added to every value of a `realistic_row`.
pub fn mean_shifted_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mean = rng.sign() * (1.0 + 2.0 * rng.uniform()) * 0.8;
    realistic_row(rng, n).into_iter().map(|v| v + mean).collect()
}

/// The same with a mean of 0.2 to 0.6 times the typical magnitude.
pub fn mildly_shifted_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mean = rng.sign() * (0.2 + 0.4 * rng.uniform()) * 0.8;
    realistic_row(rng, n).into_iter().map(|v| v + mean).collect()
}

/// Rows in which the same sixteenth of the channels (dimensions) are ten times
/// as large as the rest, as the keys of real models have them.
pub fn channel_rows(rng: &mut Rng, rows: usize, n: usize) -> Vec<Vec<f32>> {
    let loud: Vec<usize> = (0..n / 16).map(|_| rng.below(n)).collect();
    (0..rows)
        .map(|_| {
            let mut row: Vec<f32> = (0..n).map(|_| 0.8 * rng.gaussian()).collect();
            for &at in &loud {
                row[at] *= 10.0;
            }
            row
        })
        .collect()
}

/// Rows that share a fixed offset per channel, as the keys and values of real
/// models do: every row is the same vector of biases (typically 1.5 times the
/// typical magnitude, some channels several times that) plus Gaussian noise.
/// The errors a codec makes on the bias part are the same in every row if it
/// treats the rows alike, which is what averaging over positions (attention
/// over many keys, the weighted sum of values) cannot remove.
pub fn offset_rows(rng: &mut Rng, rows: usize, n: usize) -> Vec<Vec<f32>> {
    let bias: Vec<f32> = (0..n)
        .map(|_| {
            let g = rng.gaussian();
            1.2 * g * if rng.below(16) == 0 { 3.0 } else { 1.0 }
        })
        .collect();
    (0..rows).map(|_| bias.iter().map(|b| b + 0.8 * rng.gaussian()).collect()).collect()
}

/// Rows whose magnitude changes by orders of magnitude from group to group.
pub fn wide_range_rows(rng: &mut Rng, rows: usize, n: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| {
            let mut row = Vec::with_capacity(n);
            for _ in 0..n / GROUP {
                let magnitude = 10f32.powi(rng.below(14) as i32 - 9);
                row.extend((0..GROUP).map(|_| magnitude * rng.gaussian()));
            }
            row
        })
        .collect()
}

/// `n_rows` rows, each made by `make`.
pub fn rows(
    rng: &mut Rng,
    n_rows: usize,
    n: usize,
    make: fn(&mut Rng, usize) -> Vec<f32>,
) -> Vec<Vec<f32>> {
    (0..n_rows).map(|_| make(rng, n)).collect()
}
