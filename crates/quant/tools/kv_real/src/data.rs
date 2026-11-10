//! Loading of the captured tensors (see capture/capture.cpp).
//!
//! Layout on disk, all f32 little endian: `Lxx_q.f32` is `[token][n_head][hd]`,
//! `Lxx_k.f32` and `Lxx_v.f32` are `[token][n_kv][hd]`, `Lxx_attn.f32` is the
//! model's own attention output before the gate `[token][n_head * hd]`.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
}

impl Dims {
    pub fn group(&self) -> usize {
        self.n_head / self.n_kv
    }
}

pub struct Layer {
    pub il: usize,
    pub n: usize,
    pub dims: Dims,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Option<Vec<f32>>,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn list_layers(dir: &Path) -> Vec<usize> {
    let mut layers: Vec<usize> = fs::read_dir(dir)
        .expect("data dir")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let rest = name.strip_prefix('L')?.strip_suffix("_k.f32")?;
            rest.parse().ok()
        })
        .collect();
    layers.sort();
    layers
}

pub fn load_layer(dir: &Path, il: usize, dims: Dims) -> Layer {
    let p = |kind: &str| -> PathBuf { dir.join(format!("L{il:02}_{kind}.f32")) };
    let k = read_f32(&p("k"));
    let n = k.len() / (dims.n_kv * dims.hd);
    let q = read_f32(&p("q"));
    let v = read_f32(&p("v"));
    assert_eq!(q.len(), n * dims.n_head * dims.hd, "q size");
    assert_eq!(v.len(), k.len(), "v size");
    let attn = p("attn").exists().then(|| read_f32(&p("attn")));
    Layer { il, n, dims, q, k, v, attn }
}

impl Layer {
    /// K or V rows of one KV head, `[n][hd]`.
    pub fn kv_head(&self, data: &[f32], hk: usize) -> Vec<f32> {
        let hd = self.dims.hd;
        let mut out = Vec::with_capacity(self.n * hd);
        for t in 0..self.n {
            let at = (t * self.dims.n_kv + hk) * hd;
            out.extend_from_slice(&data[at..at + hd]);
        }
        out
    }

    pub fn k_head(&self, hk: usize) -> Vec<f32> {
        self.kv_head(&self.k, hk)
    }

    pub fn v_head(&self, hk: usize) -> Vec<f32> {
        self.kv_head(&self.v, hk)
    }

    /// Queries of one query head, `[n][hd]`.
    pub fn q_head(&self, h: usize) -> Vec<f32> {
        let hd = self.dims.hd;
        let mut out = Vec::with_capacity(self.n * hd);
        for t in 0..self.n {
            let at = (t * self.dims.n_head + h) * hd;
            out.extend_from_slice(&self.q[at..at + hd]);
        }
        out
    }
}

/// Segments of the text: `(label, first token, end token)`.
pub fn read_segments(path: &Path) -> Vec<(String, usize, usize)> {
    let Ok(text) = fs::read_to_string(path) else { return Vec::new() };
    text.lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            Some((f.next()?.to_string(), f.next()?.parse().ok()?, f.next()?.parse().ok()?))
        })
        .collect()
}
