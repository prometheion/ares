//! A writer for small made-up models of the `qwen4exp` architecture
//! (`docs/reference/qwen4exp-architecture.md`, section 8.3), so that the
//! architecture can be checked without the 63 GiB file: by the tests of this
//! crate, by GPU implementations against the reference, and by the reference
//! against the llama.cpp fork, which reads the same file.
//!
//! A model is fully determined by its [`Shape`], a seed and the tensor types.
//! The values of a tensor depend only on the seed and the tensor's name, are
//! drawn with integer arithmetic and a few multiplications, and are therefore
//! the same on every machine. Scales are chosen so that activations stay
//! finite over many layers and so that router logits and block scores are
//! spread out: a random router packs its logits so closely that two
//! implementations pick different experts.
//!
//! The file carries every tensor of section 2.3 and the metadata of 2.2, with
//! a placeholder vocabulary so that llama.cpp loads it.

use std::path::Path;

use ares_core::GgmlType;

/// The n-gram embedding of a made-up model.
#[derive(Clone, Debug, PartialEq)]
pub struct NgramShape {
    /// The layer that applies it.
    pub layer: usize,
    /// Tokens hashed together at most.
    pub ngram_size: usize,
    /// Table rows per n-gram order.
    pub heads_per_ngram: usize,
    /// Taps of the convolution.
    pub conv_kernel: usize,
    /// The token that cuts the hash window.
    pub eos_token: u32,
    /// The token an image cell hashes as.
    pub image_token: u32,
    /// Rows of each head, `(ngram_size - 1) * heads_per_ngram` values.
    pub head_vocab: Vec<u64>,
    /// Rows of the table that no head addresses (the real table has 90).
    pub spare_rows: u64,
}

/// The sizes of a made-up model; the names are those of
/// [`Config`](crate::qwen4exp::Config).
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    /// Layers.
    pub n_layer: usize,
    /// The width of one residual stream.
    pub n_embd: usize,
    /// Residual streams.
    pub n_stream: usize,
    /// The rank of the stream mixer.
    pub mixer_rank: usize,
    /// Tokens.
    pub n_vocab: usize,
    /// The trained context.
    pub n_ctx: usize,
    /// Query heads of the attention layers.
    pub n_head: usize,
    /// Key and value heads.
    pub n_head_kv: usize,
    /// The width of an attention head.
    pub head_dim: usize,
    /// Rotated values of a head.
    pub rope_dim: usize,
    /// Rotated pairs per position stream.
    pub rope_sections: [u32; 4],
    /// The rotary base.
    pub rope_base: f32,
    /// Every `interval`-th layer is an attention layer.
    pub interval: usize,
    /// Indexer query heads.
    pub indexer_heads: usize,
    /// The width of an indexer key.
    pub indexer_dim: usize,
    /// Cells per indexer block.
    pub block: usize,
    /// The selection budget in cells.
    pub top_k: usize,
    /// Taps of the recurrent layers' convolution.
    pub conv_kernel: usize,
    /// Key heads of the recurrent layers.
    pub n_k_heads: usize,
    /// Value heads of the recurrent layers.
    pub n_v_heads: usize,
    /// The width of a recurrent head.
    pub state_size: usize,
    /// Routed experts per layer.
    pub n_expert: usize,
    /// Experts a token uses.
    pub n_expert_used: usize,
    /// The inner width of a routed expert.
    pub expert_ff: usize,
    /// The inner width of the shared expert.
    pub shared_ff: usize,
    /// The n-gram embedding, when the model has one.
    pub ngram: Option<NgramShape>,
}

impl Shape {
    /// The shape the tests use. It exercises every rule section 8.3 lists:
    /// eight layers, two of them attention; an indexer budget of two blocks;
    /// three value heads per key head; two query heads per key/value head;
    /// four times as many experts as a token uses; three streams, so that a
    /// division by the stream count is not a division by a power of two; an
    /// n-gram layer with a small table and the real hash multipliers.
    pub fn small() -> Self {
        Self {
            n_layer: 8,
            n_embd: 32,
            n_stream: 3,
            mixer_rank: 5,
            n_vocab: 96,
            n_ctx: 4096,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 12,
            rope_dim: 8,
            rope_sections: [2, 1, 1, 0],
            rope_base: 10000.0,
            interval: 4,
            indexer_heads: 3,
            indexer_dim: 10,
            block: 4,
            top_k: 8,
            conv_kernel: 4,
            n_k_heads: 2,
            n_v_heads: 6,
            state_size: 6,
            n_expert: 12,
            n_expert_used: 3,
            expert_ff: 10,
            shared_ff: 14,
            ngram: Some(NgramShape {
                layer: 1,
                ngram_size: 3,
                heads_per_ngram: 2,
                conv_kernel: 4,
                eos_token: 7,
                image_token: 9,
                head_vocab: vec![101, 103, 107, 109],
                spare_rows: 4,
            }),
        }
    }

    /// A sequence of `n` tokens that depends on `seed`, with the cut token of
    /// the n-gram hash at every thirteenth place (from the sixth on), so that
    /// the cut of the hash window occurs.
    pub fn tokens(&self, n: usize, seed: u64) -> Vec<u32> {
        let cut = self.ngram.as_ref().map(|g| g.eos_token);
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match cut {
                    Some(cut) if i % 13 == 5 => cut,
                    _ => (state % self.n_vocab as u64) as u32,
                }
            })
            .collect()
    }

    fn wide(&self) -> usize {
        self.n_stream * self.n_embd
    }

    fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.interval)
    }
}

/// The element types of a made-up model's tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Types {
    /// Everything `f32`. With it llama.cpp's CPU backend rounds no
    /// activation, so what remains between it and the reference is rounding.
    F32,
    /// The unquantized types of the real file: BF16 for the mixers, the
    /// indexer projections, `ssm_alpha`, `ssm_beta`, `ple_key` and
    /// `ple_value`, F16 for `ple_conv1d`, `f32` for everything else (which is
    /// quantized in the real file).
    Shipped,
}

/// A made-up model.
#[derive(Clone, Debug, PartialEq)]
pub struct Synthetic {
    /// Its sizes.
    pub shape: Shape,
    /// The seed of its values.
    pub seed: u64,
    /// The element types of its tensors.
    pub types: Types,
    /// The type of the routed experts and of the n-gram table when it is not
    /// `f32`: any type [`ares_quant::quantize`] writes, or IQ4_NL (the type
    /// of the real n-gram table), whose block divides the row widths. `None`
    /// leaves them `f32`.
    pub lazy_type: Option<GgmlType>,
}

impl Synthetic {
    /// The model of [`Shape::small`] with `seed`, all `f32`.
    pub fn small(seed: u64) -> Self {
        Self { shape: Shape::small(), seed, types: Types::F32, lazy_type: None }
    }

    /// The same model with other tensor types.
    pub fn types(mut self, types: Types) -> Self {
        self.types = types;
        self
    }

    /// The bytes of the GGUF file.
    ///
    /// # Panics
    ///
    /// Panics when [`Synthetic::lazy_type`] names a type that cannot hold the
    /// rows of this shape.
    pub fn bytes(&self) -> Vec<u8> {
        let mut file = File::default();
        self.metadata(&mut file);
        self.tensors(&mut file);
        file.build()
    }

    /// Writes the GGUF file to `path`.
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.bytes())
    }

    fn metadata(&self, file: &mut File) {
        let s = &self.shape;
        file.string("general.architecture", "qwen4exp");
        file.string("general.name", "ares synthetic qwen4exp");
        let u = |file: &mut File, key: &str, value: usize| {
            file.u32(&format!("qwen4exp.{key}"), value as u32);
        };
        u(file, "block_count", s.n_layer);
        u(file, "context_length", s.n_ctx);
        u(file, "embedding_length", s.n_embd);
        u(file, "attention.head_count", s.n_head);
        u(file, "attention.head_count_kv", s.n_head_kv);
        u(file, "attention.key_length", s.head_dim);
        u(file, "attention.value_length", s.head_dim);
        file.f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6);
        u(file, "rope.dimension_count", s.rope_dim);
        file.u32_array("qwen4exp.rope.dimension_sections", &s.rope_sections);
        file.f32("qwen4exp.rope.freq_base", s.rope_base);
        u(file, "full_attention_interval", s.interval);
        let attention: Vec<bool> = (0..s.n_layer).map(|l| s.is_attention(l)).collect();
        file.bool_array(
            "qwen4exp.attention.recurrent_layers",
            &attention.iter().map(|a| !a).collect::<Vec<_>>(),
        );
        file.u32_array(
            "qwen4exp.attention.compress_ratios",
            &attention.iter().map(|&a| if a { s.block as u32 } else { 0 }).collect::<Vec<_>>(),
        );
        u(file, "attention.indexer.head_count", s.indexer_heads);
        u(file, "attention.indexer.key_length", s.indexer_dim);
        u(file, "attention.indexer.top_k", s.top_k);
        u(file, "ssm.state_size", s.state_size);
        u(file, "ssm.group_count", s.n_k_heads);
        u(file, "ssm.time_step_rank", s.n_v_heads);
        u(file, "ssm.inner_size", s.n_v_heads * s.state_size);
        u(file, "ssm.conv_kernel", s.conv_kernel);
        u(file, "hyper_connection.count", s.n_stream);
        u(file, "hyper_connection.low_rank", s.mixer_rank);
        u(file, "expert_count", s.n_expert);
        u(file, "expert_used_count", s.n_expert_used);
        u(file, "expert_feed_forward_length", s.expert_ff);
        u(file, "expert_shared_feed_forward_length", s.shared_ff);
        if let Some(g) = &s.ngram {
            let heads = (g.ngram_size - 1) * g.heads_per_ngram;
            file.u32_array("qwen4exp.ple.layers", &[g.layer as u32]);
            u(file, "ple.ngram_size", g.ngram_size);
            u(file, "ple.heads_per_ngram", g.heads_per_ngram);
            u(file, "ple.conv_kernel", g.conv_kernel);
            file.u32("qwen4exp.ple.eos_token_id", g.eos_token);
            file.u32("qwen4exp.ple.image_token_id", g.image_token);
            u(file, "embedding_length_per_layer_input", s.n_embd / heads);
            file.u64_array("qwen4exp.ple.layer_multipliers", &MULTIPLIERS[..g.ngram_size]);
            file.u64_array("qwen4exp.ple.head_vocab_sizes", &g.head_vocab);
            let mut at = 0;
            let offsets: Vec<u64> = g
                .head_vocab
                .iter()
                .map(|size| {
                    at += size;
                    at - size
                })
                .collect();
            file.u64_array("qwen4exp.ple.head_offsets", &offsets);
        }
        let tokens: Vec<String> = (0..s.n_vocab).map(|i| format!("t{i}")).collect();
        file.string("tokenizer.ggml.model", "gpt2");
        file.string_array("tokenizer.ggml.tokens", &tokens);
        file.i32_array("tokenizer.ggml.token_type", &vec![1; s.n_vocab]);
        file.string_array("tokenizer.ggml.merges", &[]);
    }

    fn tensors(&self, file: &mut File) {
        let s = &self.shape;
        let (n, wide, lr) = (s.n_embd, s.wide(), s.mixer_rank);
        let shipped = self.types == Types::Shipped;
        let bf16 = if shipped { GgmlType::BF16 } else { GgmlType::F32 };
        let lazy = self.lazy_type.unwrap_or(GgmlType::F32);
        let f32t = GgmlType::F32;
        // A matrix `[cols, rows]` whose outputs have about `gain` times the
        // RMS of its inputs.
        let matrix = |file: &mut File, name: &str, dims: &[usize], gain: f32, ty: GgmlType| {
            let scale = gain / (dims[0] as f32).sqrt();
            let values = self.normal(name, dims.iter().product(), scale, 0.0);
            file.tensor(name, dims, ty, &values);
        };

        matrix(file, "token_embd.weight", &[n, s.n_vocab], (n as f32).sqrt(), f32t);
        matrix(file, "output.weight", &[n, s.n_vocab], 3.0, f32t);
        matrix(file, "output_hc_down.weight", &[wide, lr], 2.0, bf16);
        matrix(file, "output_hc_up.weight", &[lr, wide], 2.0, bf16);
        if let Some(g) = &s.ngram {
            let heads = (g.ngram_size - 1) * g.heads_per_ngram;
            let rows = (g.head_vocab.iter().sum::<u64>() + g.spare_rows) as usize;
            let dim = n / heads;
            matrix(file, "per_layer_token_embd.weight", &[dim, rows], (dim as f32).sqrt(), lazy);
        }
        for layer in 0..s.n_layer {
            let name = |what: &str| format!("blk.{layer}.{what}");
            for which in ["attn", "ffn"] {
                matrix(file, &name(&format!("hc_{which}_down.weight")), &[wide, lr], 2.0, bf16);
                matrix(file, &name(&format!("hc_{which}_up.weight")), &[lr, wide], 2.0, bf16);
                matrix(
                    file,
                    &name(&format!("hc_{which}_inject.weight")),
                    &[wide, s.n_stream],
                    2.0,
                    bf16,
                );
            }
            // The router's logits are spread over several units.
            matrix(file, &name("ffn_gate_inp.weight"), &[n, s.n_expert], 6.0, f32t);
            matrix(file, &name("ffn_gate_exps.weight"), &[n, s.expert_ff, s.n_expert], 1.5, lazy);
            matrix(file, &name("ffn_up_exps.weight"), &[n, s.expert_ff, s.n_expert], 1.5, lazy);
            matrix(file, &name("ffn_down_exps.weight"), &[s.expert_ff, n, s.n_expert], 1.0, lazy);
            matrix(file, &name("ffn_gate_inp_shexp.weight"), &[n], 2.0, f32t);
            matrix(file, &name("ffn_gate_shexp.weight"), &[n, s.shared_ff], 1.5, f32t);
            matrix(file, &name("ffn_up_shexp.weight"), &[n, s.shared_ff], 1.5, f32t);
            matrix(file, &name("ffn_down_shexp.weight"), &[s.shared_ff, n], 1.0, f32t);
            if s.is_attention(layer) {
                let (q_dim, kv_dim) = (s.n_head * s.head_dim, s.n_head_kv * s.head_dim);
                matrix(file, &name("attn_q.weight"), &[n, q_dim * 2], 1.5, f32t);
                matrix(file, &name("attn_k.weight"), &[n, kv_dim], 1.5, f32t);
                matrix(file, &name("attn_v.weight"), &[n, kv_dim], 1.5, f32t);
                matrix(file, &name("attn_output.weight"), &[q_dim, n], 1.5, f32t);
                let idx = s.indexer_dim;
                matrix(
                    file,
                    &name("indexer.q_proj.weight"),
                    &[n, s.indexer_heads * idx],
                    1.5,
                    bf16,
                );
                matrix(file, &name("indexer.k_proj.weight"), &[n, idx], 1.5, bf16);
            } else {
                let (kd, vd) = (s.n_k_heads * s.state_size, s.n_v_heads * s.state_size);
                matrix(file, &name("attn_qkv.weight"), &[n, 2 * kd + vd], 1.5, f32t);
                matrix(file, &name("attn_gate.weight"), &[n, vd], 1.5, f32t);
                matrix(file, &name("ssm_alpha.weight"), &[n, s.n_v_heads], 2.0, bf16);
                matrix(file, &name("ssm_beta.weight"), &[n, s.n_v_heads], 2.0, bf16);
                matrix(file, &name("ssm_out.weight"), &[vd, n], 1.5, f32t);
            }
            if s.ngram.as_ref().is_some_and(|g| g.layer == layer) {
                matrix(file, &name("ple_key.weight"), &[n, wide], 1.5, bf16);
                matrix(file, &name("ple_value.weight"), &[n, n], 1.5, bf16);
            }
        }

        // Vectors: norm weights hold `1 + w`, attention norms a larger weight
        // so that the softmax is not flat.
        let vector =
            |file: &mut File, name: &str, dims: &[usize], scale: f32, mean: f32, ty: GgmlType| {
                let values = self.normal(name, dims.iter().product(), scale, mean);
                file.tensor(name, dims, ty, &values);
            };
        vector(file, "output_hc_norm.weight", &[wide], 0.1, 1.0, f32t);
        for layer in 0..s.n_layer {
            let name = |what: &str| format!("blk.{layer}.{what}");
            vector(file, &name("hc_attn_norm.weight"), &[wide], 0.1, 1.0, f32t);
            vector(file, &name("hc_ffn_norm.weight"), &[wide], 0.1, 1.0, f32t);
            if s.is_attention(layer) {
                vector(file, &name("attn_q_norm.weight"), &[s.head_dim], 0.2, 1.5, f32t);
                vector(file, &name("attn_k_norm.weight"), &[s.head_dim], 0.2, 1.5, f32t);
                vector(file, &name("indexer.q_norm.weight"), &[s.indexer_dim], 0.1, 1.0, f32t);
                vector(file, &name("indexer.k_norm.weight"), &[s.indexer_dim], 0.1, 1.0, f32t);
            } else {
                let cg = (2 * s.n_k_heads + s.n_v_heads) * s.state_size;
                vector(file, &name("ssm_conv1d.weight"), &[s.conv_kernel, cg], 0.5, 0.0, f32t);
                vector(file, &name("ssm_dt.bias"), &[s.n_v_heads], 1.0, 0.0, f32t);
                vector(file, &name("ssm_norm.weight"), &[s.state_size], 0.1, 1.0, f32t);
                // ssm_a = -exp(A_log): negative, over two orders of magnitude.
                let a: Vec<f32> = self
                    .normal(&name("ssm_a"), s.n_v_heads, 1.2, -1.0)
                    .iter()
                    .map(|x| -x.exp())
                    .collect();
                file.tensor(&name("ssm_a"), &[s.n_v_heads], f32t, &a);
            }
            if let Some(g) = s.ngram.as_ref().filter(|g| g.layer == layer) {
                let f16 = if shipped { GgmlType::F16 } else { GgmlType::F32 };
                vector(file, &name("ple_norm_key.weight"), &[wide], 0.1, 1.0, f32t);
                vector(file, &name("ple_norm_query.weight"), &[wide], 0.1, 1.0, f32t);
                vector(file, &name("ple_norm_conv.weight"), &[wide], 0.1, 1.0, f32t);
                vector(file, &name("ple_conv1d.weight"), &[g.conv_kernel, wide], 0.5, 0.0, f16);
            }
        }
    }

    /// `count` values of mean `mean` and standard deviation `scale`, bounded
    /// by `2 * sqrt(3) * scale` around the mean, that depend only on the seed
    /// and `name`: each is the sum of four uniform numbers, which needs no
    /// function of the math library.
    fn normal(&self, name: &str, count: usize, scale: f32, mean: f32) -> Vec<f32> {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in name.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let mut state = hash ^ self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut uniform = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            // 24 random bits as a number in [-0.5, 0.5).
            (z >> 40) as f32 / 16_777_216.0 - 0.5
        };
        (0..count)
            .map(|_| {
                let sum = uniform() + uniform() + uniform() + uniform();
                mean + sum * 3.0f32.sqrt() * scale
            })
            .collect()
    }
}

/// The hash multipliers of Qwen3.8-Flash-Next.
const MULTIPLIERS: [u64; 3] = [23703573157769, 20109073645365, 8052911324071];

const ALIGNMENT: usize = 32;

/// A GGUF version 3 file under construction. (`ares_gguf::GgufBuilder` has no
/// arrays of 64-bit integers or booleans, which this architecture's metadata
/// needs.)
#[derive(Default)]
struct File {
    meta: Vec<u8>,
    n_meta: u64,
    tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
}

impl File {
    fn put_str(out: &mut Vec<u8>, text: &[u8]) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text);
    }

    fn key(&mut self, key: &str, type_id: u32) {
        Self::put_str(&mut self.meta, key.as_bytes());
        self.meta.extend_from_slice(&type_id.to_le_bytes());
        self.n_meta += 1;
    }

    fn array(&mut self, key: &str, elem_type: u32, len: usize) {
        self.key(key, 9);
        self.meta.extend_from_slice(&elem_type.to_le_bytes());
        self.meta.extend_from_slice(&(len as u64).to_le_bytes());
    }

    fn u32(&mut self, key: &str, value: u32) {
        self.key(key, 4);
        self.meta.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, key: &str, value: f32) {
        self.key(key, 6);
        self.meta.extend_from_slice(&value.to_le_bytes());
    }

    fn string(&mut self, key: &str, value: &str) {
        self.key(key, 8);
        Self::put_str(&mut self.meta, value.as_bytes());
    }

    fn u32_array(&mut self, key: &str, values: &[u32]) {
        self.array(key, 4, values.len());
        for value in values {
            self.meta.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn i32_array(&mut self, key: &str, values: &[i32]) {
        self.array(key, 5, values.len());
        for value in values {
            self.meta.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn u64_array(&mut self, key: &str, values: &[u64]) {
        self.array(key, 10, values.len());
        for value in values {
            self.meta.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn bool_array(&mut self, key: &str, values: &[bool]) {
        self.array(key, 7, values.len());
        self.meta.extend(values.iter().map(|&b| u8::from(b)));
    }

    fn string_array(&mut self, key: &str, values: &[String]) {
        self.array(key, 8, values.len());
        for value in values {
            Self::put_str(&mut self.meta, value.as_bytes());
        }
    }

    /// Adds a tensor of `values` in GGUF dimension order, encoded as `ty`.
    fn tensor(&mut self, name: &str, dims: &[usize], ty: GgmlType, values: &[f32]) {
        let data = match ty {
            GgmlType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            GgmlType::F16 => {
                values.iter().flat_map(|&v| ares_quant::f32_to_f16(v).to_le_bytes()).collect()
            }
            GgmlType::BF16 => {
                values.iter().flat_map(|&v| ares_quant::f32_to_bf16(v).to_le_bytes()).collect()
            }
            GgmlType::IQ4_NL => iq4_nl_rows(dims[0], values),
            other => {
                // Row by row, so that every row is whole blocks.
                let (cols, mut data) = (dims[0], Vec::new());
                let row_bytes = other
                    .bytes_for(cols as u64)
                    .unwrap_or_else(|| panic!("{other} cannot hold rows of {cols} values"));
                for row in values.chunks_exact(cols) {
                    let at = data.len();
                    data.resize(at + row_bytes as usize, 0u8);
                    ares_quant::quantize(other, row, &mut data[at..])
                        .unwrap_or_else(|e| panic!("tensor '{name}' as {other}: {e}"));
                }
                data
            }
        };
        let dims = dims.iter().map(|&d| d as u64).collect();
        self.tensors.push((name.to_owned(), dims, ty, data));
    }

    fn build(&self) -> Vec<u8> {
        let pad = |out: &mut Vec<u8>| out.resize(out.len().next_multiple_of(ALIGNMENT), 0);
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.n_meta.to_le_bytes());
        out.extend_from_slice(&self.meta);
        let mut offset = 0usize;
        for (name, dims, ty, data) in &self.tensors {
            Self::put_str(&mut out, name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&ty.id().to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset = (offset + data.len()).next_multiple_of(ALIGNMENT);
        }
        pad(&mut out);
        for (_, _, _, data) in &self.tensors {
            out.extend_from_slice(data);
            pad(&mut out);
        }
        out
    }
}

/// Rows of `cols` values as IQ4_NL blocks: per block of 32 values an f16 scale
/// that maps the largest magnitude to the end of the code table, and for every
/// value the nearest of the 16 codes. [`ares_quant::quantize`] has no encoder
/// for this type, and a made-up table needs valid blocks, not the best ones.
fn iq4_nl_rows(cols: usize, values: &[f32]) -> Vec<u8> {
    const CODES: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];
    assert!(cols.is_multiple_of(32), "iq4_nl cannot hold rows of {cols} values");
    let mut data = Vec::with_capacity(values.len() / 32 * 18);
    for block in values.as_chunks::<32>().0 {
        let largest = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let stored = ares_quant::f32_to_f16(largest / 127.0);
        let scale = ares_quant::f16_to_f32(stored);
        let code = |v: f32| -> u8 {
            let nearest = CODES
                .iter()
                .enumerate()
                .min_by(|a, b| (v - scale * a.1).abs().total_cmp(&(v - scale * b.1).abs()));
            nearest.map_or(0, |(i, _)| i as u8)
        };
        data.extend_from_slice(&stored.to_le_bytes());
        data.extend((0..16).map(|j| code(block[j]) | code(block[j + 16]) << 4));
    }
    data
}
