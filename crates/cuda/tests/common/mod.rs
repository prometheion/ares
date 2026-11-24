//! What the tests of this crate share: the devices they open, a small generator, the copies
//! of values to and from a device, and weight blocks of every type ([`blocks`]).

// Each test crate uses what it needs of this module.
#![allow(dead_code)]

pub mod blocks;

use std::sync::OnceLock;

use ares_cuda::{Device, DeviceBuffer, Wait, device_count};

/// Every device of the machine, opened once for all the tests of a file. `None`, and a line
/// that says so, unless `ARES_TEST_CUDA` is set: opening a device takes a few hundred MiB of
/// its memory, which is not to be taken next to a model that fills the cards.
pub fn gpus() -> Option<&'static [Device]> {
    static DEVICES: OnceLock<Option<Vec<Device>>> = OnceLock::new();
    DEVICES
        .get_or_init(|| {
            if std::env::var_os("ARES_TEST_CUDA").is_none() {
                eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
                return None;
            }
            let count = device_count().expect("the CUDA driver");
            assert!(count > 0, "ARES_TEST_CUDA is set but the driver reports no device");
            Some((0..count).map(|i| Device::open(i, Wait::Block).expect("open device")).collect())
        })
        .as_deref()
}

/// A small generator with a fixed sequence (xorshift), so that failures reproduce.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A value of a normal distribution (Box-Muller), occasionally an outlier: real
    /// activations have a few much larger values per row.
    pub fn activation(&mut self) -> f32 {
        let u1 = (self.unit() + 1.0) * 0.5 + 1e-7;
        let u2 = self.unit();
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::PI * u2).cos();
        if self.next().is_multiple_of(400) { normal * 30.0 } else { normal }
    }

    /// A number below `n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// `n` values in [-scale, scale).
    pub fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

/// The bytes of f32 values.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The bytes of u32 values.
pub fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The f32 values of bytes.
pub fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// The first `len` bytes of a device buffer.
pub fn download(buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    buffer.read(0, &mut bytes).unwrap();
    bytes
}

/// The first `n` f32 values of a device buffer.
pub fn read_f32(buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
    bytes_f32(&download(buffer, n * 4))
}

/// A device buffer that holds `bytes` (at least 16 bytes, for an empty list).
pub fn upload(device: &Device, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, (bytes.len() as u64).max(16)).unwrap();
    if !bytes.is_empty() {
        buffer.write(0, bytes).unwrap();
    }
    buffer
}

/// A device buffer that holds `values` (at least 16 bytes, for an empty list).
pub fn upload_f32(device: &Device, values: &[f32]) -> DeviceBuffer {
    upload(device, &f32_bytes(values))
}

/// Largest absolute difference, relative to the largest value of `reference`.
pub fn rel_max(got: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(got.len(), reference.len());
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(reference).fold(0.0f64, |m, (g, r)| m.max((f64::from(*g) - r).abs())) / scale
}
