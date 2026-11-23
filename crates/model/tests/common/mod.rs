//! A small `qwen35` model with made-up weights, written with `GgufBuilder`, and
//! an independent implementation of its forward pass.
//!
//! The independent implementation shares nothing with the crate under test
//! except the dequantizer: it takes the weights as the plain arrays they were
//! generated as, recomputes whole sequences in `f64` with the formulas of the
//! architecture document written out literally (dense matrices, a zero-padded
//! convolution, a recurrence from the empty state), and keeps no state.
//!
//! Beside it: what the tests that open GPUs share (`gpu`) and the small
//! helpers of all tests (`helpers`).

#![allow(dead_code)]

pub mod gpu;
pub mod helpers;

use std::path::PathBuf;

use ares_core::GgmlType;
use ares_gguf::GgufBuilder;

/// The shape of a made-up model. Defaults give eight blocks (blocks 3 and 7 are
/// attention), three value heads per key head and two query heads per
/// key/value head, which is the grouping of the 27B model.
#[derive(Clone, Debug)]
pub struct Spec {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub sections: [u32; 4],
    pub rope_base: f32,
    pub eps: f32,
    pub interval: usize,
    pub conv_kernel: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub state: usize,
    /// Adds the next-token-prediction block after the model's blocks.
    pub nextn: bool,
    /// Leaves out `output.weight`, so the head is `token_embd.weight`.
    pub tied: bool,
    /// How the matrices are stored: `F32` or `Q8_0`.
    pub matrix_type: GgmlType,
    pub seed: u64,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            n_layer: 8,
            n_embd: 64,
            n_ff: 96,
            n_vocab: 128,
            n_ctx: 512,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 32,
            rope_dim: 16,
            sections: [3, 3, 2, 0],
            rope_base: 10_000_000.0,
            eps: 1e-6,
            interval: 4,
            conv_kernel: 4,
            k_heads: 2,
            v_heads: 6,
            state: 16,
            nextn: false,
            tied: false,
            matrix_type: GgmlType::F32,
            seed: 7,
        }
    }
}

impl Spec {
    pub fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.interval)
    }
    pub fn key_dim(&self) -> usize {
        self.k_heads * self.state
    }
    pub fn value_dim(&self) -> usize {
        self.v_heads * self.state
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// One tensor as generated: its GGUF dimensions and the values it holds when
/// decoded.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub values: Vec<f32>,
    pub ty: GgmlType,
}

/// A generated model.
#[derive(Clone, Debug)]
pub struct Synth {
    pub spec: Spec,
    pub tensors: Vec<Tensor>,
    /// The number of tokens the vocabulary lists, when it is not `n_vocab`.
    pub listed_tokens: Option<usize>,
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

impl Synth {
    pub fn new(spec: Spec) -> Self {
        let mut rng = Rng(spec.seed);
        let mut tensors = Vec::new();
        let s = &spec;
        let matrix = |rng: &mut Rng,
                      tensors: &mut Vec<Tensor>,
                      name: String,
                      cols: usize,
                      rows: usize,
                      gain: f64| {
            let sigma = gain / (cols as f64).sqrt();
            let values: Vec<f32> =
                (0..cols * rows).map(|_| (rng.normal() * sigma) as f32).collect();
            let (ty, values) = if s.matrix_type == GgmlType::Q8_0 {
                // Store as Q8_0 and keep what it decodes to.
                let mut bytes =
                    vec![0u8; GgmlType::Q8_0.bytes_for(values.len() as u64).unwrap() as usize];
                ares_quant::quantize(GgmlType::Q8_0, &values, &mut bytes).unwrap();
                let mut decoded = vec![0.0f32; values.len()];
                ares_quant::dequantize(GgmlType::Q8_0, &bytes, &mut decoded).unwrap();
                (GgmlType::Q8_0, decoded)
            } else {
                (GgmlType::F32, values)
            };
            tensors.push(Tensor { name, dims: vec![cols as u64, rows as u64], values, ty });
        };
        let vector = |rng: &mut Rng,
                      tensors: &mut Vec<Tensor>,
                      name: String,
                      dims: Vec<u64>,
                      f: &dyn Fn(&mut Rng) -> f64| {
            let n: u64 = dims.iter().product();
            let values: Vec<f32> = (0..n).map(|_| f(rng) as f32).collect();
            tensors.push(Tensor { name, dims, values, ty: GgmlType::F32 });
        };
        let norm = |r: &mut Rng| 1.0 + 0.1 * r.normal();

        matrix(&mut rng, &mut tensors, "token_embd.weight".into(), s.n_embd, s.n_vocab, 1.0);
        vector(&mut rng, &mut tensors, "output_norm.weight".into(), vec![s.n_embd as u64], &norm);
        if !s.tied {
            matrix(&mut rng, &mut tensors, "output.weight".into(), s.n_embd, s.n_vocab, 1.0);
        }
        let blocks = s.n_layer + usize::from(s.nextn);
        for l in 0..blocks {
            let p = |what: &str| format!("blk.{l}.{what}");
            let attention = l >= s.n_layer || s.is_attention(l);
            vector(&mut rng, &mut tensors, p("attn_norm.weight"), vec![s.n_embd as u64], &norm);
            vector(
                &mut rng,
                &mut tensors,
                p("post_attention_norm.weight"),
                vec![s.n_embd as u64],
                &norm,
            );
            matrix(&mut rng, &mut tensors, p("ffn_gate.weight"), s.n_embd, s.n_ff, 1.0);
            matrix(&mut rng, &mut tensors, p("ffn_up.weight"), s.n_embd, s.n_ff, 1.0);
            matrix(&mut rng, &mut tensors, p("ffn_down.weight"), s.n_ff, s.n_embd, 1.0);
            if attention {
                let q = s.n_head * s.head_dim;
                let kv = s.n_head_kv * s.head_dim;
                matrix(&mut rng, &mut tensors, p("attn_q.weight"), s.n_embd, q * 2, 1.5);
                matrix(&mut rng, &mut tensors, p("attn_k.weight"), s.n_embd, kv, 1.5);
                matrix(&mut rng, &mut tensors, p("attn_v.weight"), s.n_embd, kv, 1.0);
                matrix(&mut rng, &mut tensors, p("attn_output.weight"), q, s.n_embd, 1.0);
                vector(
                    &mut rng,
                    &mut tensors,
                    p("attn_q_norm.weight"),
                    vec![s.head_dim as u64],
                    &norm,
                );
                vector(
                    &mut rng,
                    &mut tensors,
                    p("attn_k_norm.weight"),
                    vec![s.head_dim as u64],
                    &norm,
                );
            } else {
                matrix(&mut rng, &mut tensors, p("attn_qkv.weight"), s.n_embd, s.conv_dim(), 1.0);
                matrix(&mut rng, &mut tensors, p("attn_gate.weight"), s.n_embd, s.value_dim(), 1.0);
                vector(
                    &mut rng,
                    &mut tensors,
                    p("ssm_conv1d.weight"),
                    vec![s.conv_kernel as u64, s.conv_dim() as u64],
                    &|r| 0.5 * r.normal(),
                );
                vector(&mut rng, &mut tensors, p("ssm_dt.bias"), vec![s.v_heads as u64], &|r| {
                    0.3 * r.normal()
                });
                vector(&mut rng, &mut tensors, p("ssm_a"), vec![s.v_heads as u64], &|r| {
                    -(0.2 + 2.0 * r.uniform())
                });
                matrix(&mut rng, &mut tensors, p("ssm_beta.weight"), s.n_embd, s.v_heads, 1.0);
                matrix(&mut rng, &mut tensors, p("ssm_alpha.weight"), s.n_embd, s.v_heads, 1.0);
                vector(&mut rng, &mut tensors, p("ssm_norm.weight"), vec![s.state as u64], &norm);
                matrix(&mut rng, &mut tensors, p("ssm_out.weight"), s.value_dim(), s.n_embd, 1.0);
            }
            if l >= s.n_layer {
                matrix(
                    &mut rng,
                    &mut tensors,
                    p("nextn.eh_proj.weight"),
                    2 * s.n_embd,
                    s.n_embd,
                    1.0,
                );
                vector(
                    &mut rng,
                    &mut tensors,
                    p("nextn.enorm.weight"),
                    vec![s.n_embd as u64],
                    &norm,
                );
                vector(
                    &mut rng,
                    &mut tensors,
                    p("nextn.hnorm.weight"),
                    vec![s.n_embd as u64],
                    &norm,
                );
                vector(
                    &mut rng,
                    &mut tensors,
                    p("nextn.shared_head_norm.weight"),
                    vec![s.n_embd as u64],
                    &norm,
                );
            }
        }
        Synth { spec, tensors, listed_tokens: None }
    }

    pub fn tensor(&self, name: &str) -> &Tensor {
        self.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no tensor {name}"))
    }

    /// The metadata as the real files have it (plus a vocabulary of the right
    /// size, so that llama.cpp can load the file too), without the key `skip`.
    pub fn metadata(&self, skip: Option<&str>) -> GgufBuilder {
        enum Kv {
            U(usize),
            F(f32),
            Sections([i32; 4]),
        }
        let s = &self.spec;
        let mut keys = vec![
            ("block_count", Kv::U(s.n_layer + usize::from(s.nextn))),
            ("context_length", Kv::U(s.n_ctx)),
            ("embedding_length", Kv::U(s.n_embd)),
            ("feed_forward_length", Kv::U(s.n_ff)),
            ("attention.head_count", Kv::U(s.n_head)),
            ("attention.head_count_kv", Kv::U(s.n_head_kv)),
            ("attention.key_length", Kv::U(s.head_dim)),
            ("attention.value_length", Kv::U(s.head_dim)),
            ("attention.layer_norm_rms_epsilon", Kv::F(s.eps)),
            ("rope.dimension_count", Kv::U(s.rope_dim)),
            ("rope.dimension_sections", Kv::Sections(s.sections.map(|n| n as i32))),
            ("rope.freq_base", Kv::F(s.rope_base)),
            ("ssm.conv_kernel", Kv::U(s.conv_kernel)),
            ("ssm.state_size", Kv::U(s.state)),
            ("ssm.group_count", Kv::U(s.k_heads)),
            ("ssm.time_step_rank", Kv::U(s.v_heads)),
            ("ssm.inner_size", Kv::U(s.value_dim())),
            ("full_attention_interval", Kv::U(s.interval)),
        ];
        if s.nextn {
            keys.push(("nextn_predict_layers", Kv::U(1)));
        }
        let mut b = GgufBuilder::new().str("general.architecture", "qwen35");
        for (key, value) in keys {
            if Some(key) == skip {
                continue;
            }
            let full = format!("qwen35.{key}");
            b = match value {
                Kv::U(v) => b.u32(&full, v as u32),
                Kv::F(v) => b.f32(&full, v),
                Kv::Sections(v) => b.i32_array(&full, &v),
            };
        }
        let listed = self.listed_tokens.unwrap_or(s.n_vocab);
        let tokens: Vec<String> = (0..listed).map(|i| format!("t{i}")).collect();
        b.str("tokenizer.ggml.model", "gpt2")
            .str_array("tokenizer.ggml.tokens", &tokens)
            .i32_array("tokenizer.ggml.token_type", &vec![1; listed])
            .str_array::<&str>("tokenizer.ggml.merges", &[])
    }

    pub fn builder(&self) -> GgufBuilder {
        let mut b = self.metadata(None);
        for t in &self.tensors {
            b = if t.ty == GgmlType::F32 {
                b.tensor_f32(&t.name, &t.dims, &t.values)
            } else {
                let mut bytes = vec![0u8; t.ty.bytes_for(t.values.len() as u64).unwrap() as usize];
                ares_quant::quantize(t.ty, &t.values, &mut bytes).unwrap();
                b.tensor(&t.name, &t.dims, t.ty, bytes)
            };
        }
        b
    }
}

/// A fresh directory under the system temp dir, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("ares-model-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// The independent implementation.
// ---------------------------------------------------------------------------

/// What the independent implementation computes for a whole sequence.
pub struct Naive {
    /// `hidden[block][row]`.
    pub hidden: Vec<Vec<Vec<f64>>>,
    /// The logits of every row.
    pub logits: Vec<Vec<f64>>,
}

fn mat(m: &Synth, name: &str) -> (usize, usize, Vec<f64>) {
    let t = m.tensor(name);
    (t.dims[0] as usize, t.dims[1] as usize, t.values.iter().map(|&v| f64::from(v)).collect())
}

fn vecs(m: &Synth, name: &str) -> Vec<f64> {
    m.tensor(name).values.iter().map(|&v| f64::from(v)).collect()
}

/// `W x` with `W` stored `[rows][cols]`.
fn mv(w: &(usize, usize, Vec<f64>), x: &[f64]) -> Vec<f64> {
    let (cols, rows, data) = w;
    assert_eq!(x.len(), *cols);
    (0..*rows).map(|r| (0..*cols).map(|c| data[r * cols + c] * x[c]).sum()).collect()
}

fn rms(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    x.iter().zip(w).map(|(v, w)| v / (mean + eps).sqrt() * w).collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

pub fn naive(m: &Synth, tokens: &[u32]) -> Naive {
    let s = &m.spec;
    let eps = f64::from(s.eps);
    let n = tokens.len();
    let embd = mat(m, "token_embd.weight");
    let mut x: Vec<Vec<f64>> = tokens
        .iter()
        .map(|&t| embd.2[t as usize * s.n_embd..(t as usize + 1) * s.n_embd].to_vec())
        .collect();
    let mut hidden = Vec::new();

    for l in 0..s.n_layer {
        let p = |what: &str| format!("blk.{l}.{what}");
        let h: Vec<Vec<f64>> =
            x.iter().map(|r| rms(r, &vecs(m, &p("attn_norm.weight")), eps)).collect();
        let mixed =
            if s.is_attention(l) { naive_attention(m, l, &h) } else { naive_recurrent(m, l, &h) };
        for (xr, mr) in x.iter_mut().zip(&mixed) {
            for (a, b) in xr.iter_mut().zip(mr) {
                *a += b;
            }
        }
        let (gate, up, down) = (
            mat(m, &p("ffn_gate.weight")),
            mat(m, &p("ffn_up.weight")),
            mat(m, &p("ffn_down.weight")),
        );
        let post = vecs(m, &p("post_attention_norm.weight"));
        for xr in x.iter_mut() {
            let h = rms(xr, &post, eps);
            let (g, u) = (mv(&gate, &h), mv(&up, &h));
            let act: Vec<f64> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            for (a, b) in xr.iter_mut().zip(mv(&down, &act)) {
                *a += b;
            }
        }
        hidden.push(x.clone());
        assert_eq!(x.len(), n);
    }

    let out_norm = vecs(m, "output_norm.weight");
    let head = if s.tied { mat(m, "token_embd.weight") } else { mat(m, "output.weight") };
    let logits = x.iter().map(|r| mv(&head, &rms(r, &out_norm, eps))).collect();
    Naive { hidden, logits }
}

fn naive_recurrent(m: &Synth, l: usize, h: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let s = &m.spec;
    let p = |what: &str| format!("blk.{l}.{what}");
    let eps = f64::from(s.eps);
    let (qkv_w, z_w) = (mat(m, &p("attn_qkv.weight")), mat(m, &p("attn_gate.weight")));
    let (beta_w, alpha_w) = (mat(m, &p("ssm_beta.weight")), mat(m, &p("ssm_alpha.weight")));
    let conv_w = vecs(m, &p("ssm_conv1d.weight"));
    let (dt_bias, a) = (vecs(m, &p("ssm_dt.bias")), vecs(m, &p("ssm_a")));
    let (norm_w, out_w) = (vecs(m, &p("ssm_norm.weight")), mat(m, &p("ssm_out.weight")));
    let (kd, vd, cd, d) = (s.key_dim(), s.value_dim(), s.conv_dim(), s.state);
    let n = h.len();

    let qkv: Vec<Vec<f64>> = h.iter().map(|r| mv(&qkv_w, r)).collect();
    // Causal convolution with zeros before the first token, then silu.
    let conv: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            (0..cd)
                .map(|c| {
                    let sum: f64 = (0..s.conv_kernel)
                        .map(|i| {
                            // tap i multiplies the input i - (kernel - 1) steps ago.
                            let back = s.conv_kernel - 1 - i;
                            if t >= back {
                                conv_w[c * s.conv_kernel + i] * qkv[t - back][c]
                            } else {
                                0.0
                            }
                        })
                        .sum();
                    silu(sum)
                })
                .collect()
        })
        .collect();

    let unit = |v: &[f64]| -> Vec<f64> {
        let norm = (v.iter().map(|x| x * x).sum::<f64>() + eps).sqrt();
        v.iter().map(|x| x / norm).collect()
    };
    // One state per value head: S[k index][v index].
    let mut state = vec![vec![vec![0.0f64; d]; d]; s.v_heads];
    let mut out = Vec::new();
    for t in 0..n {
        let z = mv(&z_w, &h[t]);
        let beta_in = mv(&beta_w, &h[t]);
        let alpha = mv(&alpha_w, &h[t]);
        let mut gated = vec![0.0f64; vd];
        for head in 0..s.v_heads {
            let kh = head % s.k_heads;
            let q = unit(&conv[t][kh * d..(kh + 1) * d]);
            let k = unit(&conv[t][kd + kh * d..kd + (kh + 1) * d]);
            let v = &conv[t][2 * kd + head * d..2 * kd + (head + 1) * d];
            let beta = sigmoid(beta_in[head]);
            let g = (alpha[head] + dt_bias[head]).exp().ln_1p() * a[head];
            let st = &mut state[head];
            for row in st.iter_mut() {
                for e in row.iter_mut() {
                    *e *= g.exp();
                }
            }
            // kv_mem[j] = sum_i S[i][j] k[i]
            let kv_mem: Vec<f64> = (0..d).map(|j| (0..d).map(|i| st[i][j] * k[i]).sum()).collect();
            let delta: Vec<f64> = (0..d).map(|j| (v[j] - kv_mem[j]) * beta).collect();
            for i in 0..d {
                for j in 0..d {
                    st[i][j] += k[i] * delta[j];
                }
            }
            let o: Vec<f64> = (0..d)
                .map(|j| (0..d).map(|i| st[i][j] * q[i]).sum::<f64>() / (d as f64).sqrt())
                .collect();
            let normed = rms(&o, &norm_w, eps);
            for j in 0..d {
                gated[head * d + j] = normed[j] * silu(z[head * d + j]);
            }
        }
        out.push(mv(&out_w, &gated));
    }
    out
}

fn naive_attention(m: &Synth, l: usize, h: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let s = &m.spec;
    let p = |what: &str| format!("blk.{l}.{what}");
    let eps = f64::from(s.eps);
    let (qw, kw, vw, ow) = (
        mat(m, &p("attn_q.weight")),
        mat(m, &p("attn_k.weight")),
        mat(m, &p("attn_v.weight")),
        mat(m, &p("attn_output.weight")),
    );
    let (qn, kn) = (vecs(m, &p("attn_q_norm.weight")), vecs(m, &p("attn_k_norm.weight")));
    let (hd, n) = (s.head_dim, h.len());
    let half = s.rope_dim / 2;
    let rope = |v: &mut Vec<f64>, pos: usize| {
        for i in 0..half {
            let angle =
                pos as f64 * f64::from(s.rope_base).powf(-2.0 * i as f64 / s.rope_dim as f64);
            let (c, sn) = (angle.cos(), angle.sin());
            let (x0, x1) = (v[i], v[i + half]);
            v[i] = x0 * c - x1 * sn;
            v[i + half] = x0 * sn + x1 * c;
        }
    };
    let mut qs = Vec::new();
    let mut gates = Vec::new();
    let mut ks = Vec::new();
    let mut vs = Vec::new();
    for (t, row) in h.iter().enumerate() {
        let qg = mv(&qw, row);
        let mut q = Vec::new();
        let mut gate = Vec::new();
        for head in 0..s.n_head {
            let chunk = &qg[head * 2 * hd..(head + 1) * 2 * hd];
            let mut qh = rms(&chunk[..hd], &qn, eps);
            rope(&mut qh, t);
            q.push(qh);
            gate.extend_from_slice(&chunk[hd..]);
        }
        let kv = mv(&kw, row);
        let k: Vec<Vec<f64>> = (0..s.n_head_kv)
            .map(|head| {
                let mut kh = rms(&kv[head * hd..(head + 1) * hd], &kn, eps);
                rope(&mut kh, t);
                kh
            })
            .collect();
        qs.push(q);
        gates.push(gate);
        ks.push(k);
        vs.push(mv(&vw, row));
    }
    let group = s.n_head / s.n_head_kv;
    (0..n)
        .map(|t| {
            let mut cat = Vec::new();
            for head in 0..s.n_head {
                let kvh = head / group;
                let scores: Vec<f64> = (0..=t)
                    .map(|j| {
                        qs[t][head].iter().zip(&ks[j][kvh]).map(|(a, b)| a * b).sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
                let total: f64 = e.iter().sum();
                for d in 0..hd {
                    let o: f64 = (0..=t).map(|j| e[j] / total * vs[j][kvh * hd + d]).sum();
                    cat.push(o * sigmoid(gates[t][head * hd + d]));
                }
            }
            mv(&ow, &cat)
        })
        .collect()
}
