//! A minimal GGUF writer.
//!
//! It exists so tests in this and other crates can build small model files
//! with exactly the metadata and tensors they need, without depending on
//! files outside the repository.

use std::io::Write;
use std::path::Path;

use ares_core::GgmlType;
use ares_core::units::align_up;

const ALIGNMENT: u64 = 32;

/// Builds a GGUF version 3 file in memory.
#[derive(Debug, Default)]
pub struct GgufBuilder {
    meta: Vec<u8>,
    n_meta: u64,
    tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(&mut self, key: &str, type_id: u32) {
        put_str(&mut self.meta, key.as_bytes());
        self.meta.extend_from_slice(&type_id.to_le_bytes());
        self.n_meta += 1;
    }

    pub fn u32(mut self, key: &str, value: u32) -> Self {
        self.key(key, 4);
        self.meta.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn u16(mut self, key: &str, value: u16) -> Self {
        self.key(key, 2);
        self.meta.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn i32(mut self, key: &str, value: i32) -> Self {
        self.key(key, 5);
        self.meta.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn u64(mut self, key: &str, value: u64) -> Self {
        self.key(key, 10);
        self.meta.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn f32(mut self, key: &str, value: f32) -> Self {
        self.key(key, 6);
        self.meta.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn bool(mut self, key: &str, value: bool) -> Self {
        self.key(key, 7);
        self.meta.push(value.into());
        self
    }

    pub fn str(mut self, key: &str, value: &str) -> Self {
        self.key(key, 8);
        put_str(&mut self.meta, value.as_bytes());
        self
    }

    /// An array of strings. Elements may be any bytes, as token texts are.
    pub fn str_array<S: AsRef<[u8]>>(mut self, key: &str, values: &[S]) -> Self {
        self.key(key, 9);
        self.meta.extend_from_slice(&8u32.to_le_bytes());
        self.meta.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            put_str(&mut self.meta, value.as_ref());
        }
        self
    }

    pub fn u32_array(mut self, key: &str, values: &[u32]) -> Self {
        self.key(key, 9);
        self.meta.extend_from_slice(&4u32.to_le_bytes());
        self.meta.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            self.meta.extend_from_slice(&value.to_le_bytes());
        }
        self
    }

    pub fn u64_array(mut self, key: &str, values: &[u64]) -> Self {
        self.key(key, 9);
        self.meta.extend_from_slice(&10u32.to_le_bytes());
        self.meta.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            self.meta.extend_from_slice(&value.to_le_bytes());
        }
        self
    }

    pub fn bool_array(mut self, key: &str, values: &[bool]) -> Self {
        self.key(key, 9);
        self.meta.extend_from_slice(&7u32.to_le_bytes());
        self.meta.extend_from_slice(&(values.len() as u64).to_le_bytes());
        self.meta.extend(values.iter().map(|&b| u8::from(b)));
        self
    }

    pub fn i32_array(mut self, key: &str, values: &[i32]) -> Self {
        self.key(key, 9);
        self.meta.extend_from_slice(&5u32.to_le_bytes());
        self.meta.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            self.meta.extend_from_slice(&value.to_le_bytes());
        }
        self
    }

    pub fn f32_array(mut self, key: &str, values: &[f32]) -> Self {
        self.key(key, 9);
        self.meta.extend_from_slice(&6u32.to_le_bytes());
        self.meta.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            self.meta.extend_from_slice(&value.to_le_bytes());
        }
        self
    }

    /// Adds a tensor. `dims` are in ggml order (innermost first) and `data`
    /// must be the encoded bytes of exactly that many values.
    ///
    /// # Panics
    ///
    /// Panics when `data` does not have the size `dims` and `ty` imply.
    pub fn tensor(mut self, name: &str, dims: &[u64], ty: GgmlType, data: Vec<u8>) -> Self {
        let n_values: u64 = dims.iter().product();
        assert_eq!(
            ty.bytes_for(n_values),
            Some(data.len() as u64),
            "tensor '{name}': data size does not match {dims:?} of {ty}"
        );
        self.tensors.push((name.to_owned(), dims.to_vec(), ty, data));
        self
    }

    /// An `f32` tensor from values.
    pub fn tensor_f32(self, name: &str, dims: &[u64], values: &[f32]) -> Self {
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensor(name, dims, GgmlType::F32, data)
    }

    /// The bytes of the file.
    pub fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.n_meta.to_le_bytes());
        out.extend_from_slice(&self.meta);

        let mut offset = 0u64;
        for (name, dims, ty, data) in &self.tensors {
            put_str(&mut out, name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&ty.id().to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            offset = align_up(offset + data.len() as u64, ALIGNMENT).expect("small test file");
        }

        pad(&mut out);
        for (_, _, _, data) in &self.tensors {
            out.extend_from_slice(data);
            pad(&mut out);
        }
        out
    }

    /// Writes the file to `path`.
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::File::create(path)?.write_all(&self.build())
    }
}

fn put_str(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn pad(out: &mut Vec<u8>) {
    let padded = align_up(out.len() as u64, ALIGNMENT).expect("small test file");
    out.resize(padded as usize, 0);
}
