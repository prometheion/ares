//! The small helpers of the tests: the files the environment names, values
//! read from bytes, how far one row of values is from another, the token a row
//! of logits picks, and a random generator.

use std::path::{Path, PathBuf};

/// The path in the variable `name`; `None` when it is not set, and a line that
/// says the test is skipped.
pub fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set {name} to run this test");
            None
        }
    }
}

/// The real 0.8B (`ARES_TEST_QWEN35_0_8B`, Qwen3.5-0.8B-Q4_K_M.gguf).
pub fn model_path() -> Option<PathBuf> {
    env_path("ARES_TEST_QWEN35_0_8B")
}

/// The `f32` values of a file, as a capture writes them.
pub fn read_f32(path: &Path) -> Vec<f32> {
    floats(&std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// `bytes` as little-endian `f32` values.
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// `|got - want| / |want|` over all values (L2).
pub fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        d += (f64::from(*g) - f64::from(*w)).powi(2);
        n += f64::from(*w).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

/// Largest absolute difference relative to the largest value of the reference.
pub fn rel_max(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(f64::from(*v).abs())).max(1e-30);
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - f64::from(*w)).abs()))
        / scale
}

/// The largest value's index (the first of equal ones).
pub fn argmax(x: &[f32]) -> u32 {
    argmax_lead(x).0
}

/// The largest value's index (the first of equal ones) and its lead over the
/// second largest.
pub fn argmax_lead(x: &[f32]) -> (u32, f32) {
    let mut best = 0;
    let mut second = f32::NEG_INFINITY;
    for i in 1..x.len() {
        if x[i] > x[best] {
            second = x[best];
            best = i;
        } else if x[i] > second {
            second = x[i];
        }
    }
    (best as u32, x[best] - second)
}

/// A xorshift generator: the same seed gives the same values.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// `n` values in `[-scale, scale)`.
    pub fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }

    /// A number below `n`.
    pub fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 20) % n
    }
}
