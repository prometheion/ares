//! The `qwen4exp` kernels of `ares-cuda` (`Q4Kernels`: stream mixer, hashed
//! n-gram block, indexer, block scores, selection and its hand-over to the
//! attention kernels) against the CPU reference of this crate, on a made-up
//! model whose sizes the kernels take (heads of 128, widths in multiples of
//! 32).
//!
//! - Every kernel against its `reference::pieces` function with the model's
//!   weights.
//! - A chain: steps of a sequence run layer by layer on the device and through
//!   [`Reference::block_observed`] from the same input, the intermediate
//!   tensors of the `Observer` compared one by one. The device runs the
//!   mixers, the n-gram block and the whole attention layer (projections,
//!   K/V append, indexer, scores, selection, attention over the lists made on
//!   the device); what belongs to other kernels is taken from the reference:
//!   the output of the recurrent part (`LayerKernels`, tested in
//!   `layers_gpu.rs`) and of the mixture of experts (`ExpertKernels`).
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-model --test qwen4exp_gpu -- --test-threads=1 --nocapture
//! ```

// The tests launch kernels on buffers they allocated a few lines earlier, with sizes taken from
// the same shapes; a comment on each of the launches would repeat that.
#![allow(clippy::undocumented_unsafe_blocks)]

use std::path::PathBuf;

use ares_core::GgmlType;
use ares_cuda::{
    AttendBuffers, AttentionKernels, AttentionShape, AttnItem, AttnRow, AttnShape, CacheFormat,
    CacheGeometry, CommitSet, Device, DeviceBuffer, DevicePtr, IndexerShape, Kernels, KeyFormat,
    LayerKernels, ListHandOver, NgramShape, Plan, Q4Kernels, RopeParams, Segment,
    SelectionRule as DeviceRule, StepDims, StepRecord, Stream,
};
use ares_gguf::Gguf;
use ares_model::qwen4exp::reference::pieces;
use ares_model::qwen4exp::synthetic::{NgramShape as SyntheticNgram, Shape, Synthetic, Types};
use ares_model::qwen4exp::weights::{AttentionWeights, Matrix, Mixer, NgramWeights, StreamMixer};
use ares_model::qwen4exp::{
    Config, LogitRows, Recorder, Reference, SelectionRule, SequenceState, Weights,
};
use common::gpu::gpus;
use common::helpers::{Rng, rel_max};

mod common;

/// A model with every part of the architecture at sizes the kernels take: three streams of
/// 256, attention heads and indexer keys of 128, a selection budget of two blocks (so that
/// rows select from their twelfth cell on), and an n-gram table of IQ4_NL rows of 32 values.
fn model() -> Synthetic {
    let shape = Shape {
        n_embd: 256,
        mixer_rank: 32,
        n_head: 8,
        n_head_kv: 2,
        head_dim: 128,
        rope_dim: 64,
        rope_sections: [11, 11, 10, 0],
        indexer_heads: 3,
        indexer_dim: 128,
        n_k_heads: 2,
        n_v_heads: 4,
        state_size: 128,
        expert_ff: 32,
        shared_ff: 32,
        ngram: Some(SyntheticNgram {
            layer: 1,
            ngram_size: 3,
            heads_per_ngram: 4,
            conv_kernel: 4,
            eos_token: 7,
            image_token: 9,
            head_vocab: vec![101, 103, 107, 109, 113, 127, 131, 137],
            spare_rows: 4,
        }),
        ..Shape::small()
    };
    Synthetic { shape, seed: 11, types: Types::Shipped, lazy_type: Some(GgmlType::IQ4_NL) }
}

fn open(name: &str, bytes: &[u8]) -> Gguf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4gpu-{name}.gguf"));
    std::fs::write(&path, bytes).expect("the test directory is writable");
    Gguf::open(&path).expect("the writer makes a valid file")
}

struct Gpu {
    device: Device,
    stream: Stream,
    q4: Q4Kernels,
    layers: LayerKernels,
    kernels: Kernels,
    attention: AttentionKernels,
}

impl Gpu {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            stream: Stream::new(device, 0).unwrap(),
            q4: Q4Kernels::load(device).unwrap(),
            layers: LayerKernels::load(device).unwrap(),
            kernels: Kernels::load(device).unwrap(),
            attention: AttentionKernels::load(device).unwrap(),
        }
    }
    fn bytes(&self, data: &[u8]) -> DeviceBuffer {
        let buffer = DeviceBuffer::new(&self.device, data.len().max(16) as u64).unwrap();
        if !data.is_empty() {
            buffer.write(0, data).unwrap();
        }
        buffer
    }
    fn floats(&self, data: &[f32]) -> DeviceBuffer {
        self.bytes(&data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
    }
    fn words(&self, data: &[u32]) -> DeviceBuffer {
        self.bytes(&data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
    }
    /// A poisoned buffer: a kernel that does not write what it should, or reads what nobody
    /// wrote, is seen.
    fn empty(&self, bytes: u64) -> DeviceBuffer {
        let buffer = DeviceBuffer::new(&self.device, bytes.max(16)).unwrap();
        buffer.write(0, &vec![0xFFu8; bytes as usize]).unwrap();
        buffer
    }
    fn zeros(&self, bytes: u64) -> DeviceBuffer {
        self.bytes(&vec![0u8; bytes as usize])
    }
    fn read_at(&self, buffer: &DeviceBuffer, offset: u64, n: usize) -> Vec<f32> {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; n * 4];
        buffer.read(offset, &mut bytes).unwrap();
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }
    fn read(&self, buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.read_at(buffer, 0, n)
    }
    fn read_words(&self, buffer: &DeviceBuffer, offset: u64, n: usize) -> Vec<u32> {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; n * 4];
        buffer.read(offset, &mut bytes).unwrap();
        bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect()
    }
}

fn report(name: &str, value: f64) {
    eprintln!("    {name}: {value:.3e}");
}

/// A matrix on the device as the file stores it.
struct DeviceMatrix {
    ty: GgmlType,
    n_in: u32,
    n_out: u32,
    w: DeviceBuffer,
}

fn device_matrix(gpu: &Gpu, m: &Matrix<'_>) -> DeviceMatrix {
    let mut bytes = Vec::new();
    for r in 0..m.rows() {
        bytes.extend_from_slice(m.row_bytes(r));
    }
    DeviceMatrix { ty: m.ty(), n_in: m.cols() as u32, n_out: m.rows() as u32, w: gpu.bytes(&bytes) }
}

/// The activations of `rows` rows of `n_in` values for the decode weight kernels: the buffer
/// and the offset of its f32 section, which is all a float matrix reads.
struct Act {
    buf: DeviceBuffer,
    f32_at: u64,
}

impl Act {
    fn new(gpu: &Gpu, n_in: usize, rows: usize) -> Self {
        let bytes = Kernels::quantized_activation_bytes(n_in as u32, rows as u32);
        Self { buf: gpu.empty(bytes), f32_at: bytes - (n_in * rows * 4) as u64 }
    }
    /// Where a kernel writes the f32 values.
    fn values(&self) -> DevicePtr {
        self.buf.ptr().offset(self.f32_at)
    }
}

/// `y = W x` for up to 8 rows with the decode kernel. The matrices of this model are float
/// types, which read the f32 section of the activations only (the rest is poison here).
fn product(gpu: &Gpu, m: &DeviceMatrix, act: &Act, rows: usize) -> DeviceBuffer {
    assert!(matches!(m.ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16), "{}", m.ty);
    let y = gpu.empty(u64::from(m.n_out) * rows as u64 * 4);
    unsafe {
        gpu.kernels
            .matvec(
                &gpu.stream,
                m.ty,
                m.w.ptr(),
                act.buf.ptr(),
                y.ptr(),
                m.n_in,
                m.n_out,
                rows as u32,
            )
            .unwrap();
    }
    y
}

struct DeviceMixer {
    norm: DeviceBuffer,
    down: DeviceMatrix,
    up: DeviceMatrix,
    inject: DeviceMatrix,
}

fn device_mixer(gpu: &Gpu, m: &StreamMixer<'_>) -> DeviceMixer {
    DeviceMixer {
        norm: gpu.floats(&m.norm),
        down: device_matrix(gpu, &m.down),
        up: device_matrix(gpu, &m.up),
        inject: device_matrix(gpu, m.inject.as_ref().expect("a layer's mixer has an inject")),
    }
}

/// The read of a mixer on the device, after the write of the block before it when `write`
/// gives that block's output and inject values. Returns the mixed input as activations and
/// the inject values.
fn read(
    gpu: &Gpu,
    c: &Config,
    mixer: &DeviceMixer,
    r: &DeviceBuffer,
    rows: usize,
    write: Option<(&DeviceBuffer, &DeviceBuffer)>,
) -> (Act, DeviceBuffer) {
    let (hc, n, rank) = (c.n_stream as u32, c.n_embd as u32, c.mixer_rank);
    let xn = Act::new(gpu, c.wide(), rows);
    let lo = Act::new(gpu, rank, rows);
    let m = Act::new(gpu, c.n_embd, rows);
    unsafe {
        match write {
            Some((out, inj)) => gpu.q4.stream_write_norm(
                &gpu.stream,
                r.ptr(),
                out.ptr(),
                inj.ptr(),
                mixer.norm.ptr(),
                xn.values(),
                rows as u32,
                hc,
                n,
                c.rms_eps,
            ),
            None => gpu.q4.stream_norm(
                &gpu.stream,
                r.ptr(),
                mixer.norm.ptr(),
                xn.values(),
                rows as u32,
                hc,
                n,
                c.rms_eps,
            ),
        }
        .unwrap();
    }
    let down = product(gpu, &mixer.down, &xn, rows);
    unsafe {
        gpu.q4.squash(&gpu.stream, down.ptr(), lo.values(), (rows * rank) as u32, hc).unwrap();
    }
    let gate = product(gpu, &mixer.up, &lo, rows);
    let inj = product(gpu, &mixer.inject, &xn, rows);
    unsafe {
        gpu.q4
            .mix_quant(&gpu.stream, xn.values(), gate.ptr(), m.buf.ptr(), rows as u32, hc, n)
            .unwrap();
    }
    (m, inj)
}

fn indexer_shape(c: &Config) -> IndexerShape {
    IndexerShape {
        n_head: c.indexer.n_head as u32,
        key_dim: c.indexer.key_dim as u32,
        block: c.indexer.block as u32,
        top_k: c.indexer.top_k as u32,
        rope_dim: c.attention.rope.dim as u32,
    }
}

fn rope_params(c: &Config) -> RopeParams {
    let r = &c.attention.rope;
    RopeParams { dim: r.dim as u32, freq_base: r.freq_base, sections: r.sections.map(|s| s as u32) }
}

fn ngram_shape(c: &Config) -> NgramShape {
    let g = c.ngram.as_ref().expect("the model has the n-gram block");
    NgramShape {
        n_stream: c.n_stream as u32,
        n_embd: c.n_embd as u32,
        heads: g.n_heads() as u32,
        row_dim: g.row_dim as u32,
        ty: g.table_type,
        conv_kernel: g.conv_kernel as u32,
        dilation: g.ngram_size as u32,
    }
}

// ---------------------------------------------------------------------------
// Each kernel against its function of the reference.

#[test]
fn every_kernel_matches_its_piece_of_the_reference() {
    let Some(gpus) = gpus() else { return };
    let gguf = open("pieces", &model().bytes());
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let (hc, n, wide, rank) = (c.n_stream, c.n_embd, c.wide(), c.mixer_rank);
    let rows = 6usize;
    for device in gpus {
        let gpu = Gpu::new(device);
        eprintln!("  {}", device.info().name);
        let mut rng = Rng(0x5eed_0000_0000_0001);

        // 3.3 with the weights of layer 2's attention mixer.
        let mixer = &w.layers[2].attn_mixer;
        let r = rng.floats(rows * wide, 2.0);
        let mut want_xn = vec![0.0f32; rows * wide];
        for (src, dst) in r.chunks_exact(wide).zip(want_xn.chunks_exact_mut(wide)) {
            pieces::stream_norm(src, &mixer.norm, n, c.rms_eps, dst);
        }
        let (r_b, norm_b) = (gpu.floats(&r), gpu.floats(&mixer.norm));
        let xn_b = gpu.empty((rows * wide * 4) as u64);
        unsafe {
            gpu.q4
                .stream_norm(
                    &gpu.stream,
                    r_b.ptr(),
                    norm_b.ptr(),
                    xn_b.ptr(),
                    rows as u32,
                    hc as u32,
                    n as u32,
                    c.rms_eps,
                )
                .unwrap();
        }
        report("stream_norm", rel_max(&gpu.read(&xn_b, rows * wide), &want_xn));

        let down = rng.floats(rows * rank, 6.0);
        let mut want_lo = vec![0.0f32; rows * rank];
        pieces::mixer_squash(&down, hc, &mut want_lo);
        let down_b = gpu.floats(&down);
        unsafe {
            gpu.q4
                .squash(&gpu.stream, down_b.ptr(), down_b.ptr(), (rows * rank) as u32, hc as u32)
                .unwrap();
        }
        report("mixer_squash", rel_max(&gpu.read(&down_b, rows * rank), &want_lo));

        let gate = rng.floats(rows * wide, 4.0);
        let mut want_m = vec![0.0f32; rows * n];
        for ((xn, gate), m) in
            want_xn.chunks_exact(wide).zip(gate.chunks_exact(wide)).zip(want_m.chunks_exact_mut(n))
        {
            pieces::mixer_mix(xn, gate, m);
        }
        let (xn_in, gate_b) = (gpu.floats(&want_xn), gpu.floats(&gate));
        let m_act = Act::new(&gpu, n, rows);
        unsafe {
            gpu.q4
                .mix_quant(
                    &gpu.stream,
                    xn_in.ptr(),
                    gate_b.ptr(),
                    m_act.buf.ptr(),
                    rows as u32,
                    hc as u32,
                    n as u32,
                )
                .unwrap();
        }
        report("mixer_mix", rel_max(&gpu.read_at(&m_act.buf, m_act.f32_at, rows * n), &want_m));

        let (out, inj) = (rng.floats(rows * n, 2.0), rng.floats(rows * hc, 5.0));
        let mut want_r = r.clone();
        for t in 0..rows {
            pieces::mixer_write(
                &mut want_r[t * wide..(t + 1) * wide],
                &out[t * n..(t + 1) * n],
                &inj[t * hc..(t + 1) * hc],
            );
        }
        let (out_b, inj_b) = (gpu.floats(&out), gpu.floats(&inj));
        unsafe {
            gpu.q4
                .stream_write(
                    &gpu.stream,
                    r_b.ptr(),
                    out_b.ptr(),
                    inj_b.ptr(),
                    rows as u32,
                    hc as u32,
                    n as u32,
                )
                .unwrap();
        }
        report("mixer_write", rel_max(&gpu.read(&r_b, rows * wide), &want_r));

        // 3.4: rows of the table, the gate, the convolution and the update.
        let ngram = c.ngram.as_ref().unwrap();
        let shape = ngram_shape(&c);
        let table = w.ngram_table.as_ref().unwrap();
        assert_eq!(table.ty(), GgmlType::IQ4_NL);
        let nw: &NgramWeights<'_> = w.layers[ngram.layer].ngram.as_ref().unwrap();
        let tokens = model().shape.tokens(rows, 5);
        let (mut fetched, mut want_emb) = (Vec::new(), vec![0.0f32; rows * n]);
        for (t, &token) in tokens.iter().enumerate() {
            let ids = pieces::ngram_rows(ngram, token, &tokens[..t]);
            for (h, &id) in ids.iter().enumerate() {
                fetched.extend_from_slice(table.row_bytes(id));
                table.decode_row(id, &mut want_emb[t * n + h * ngram.row_dim..][..ngram.row_dim]);
            }
        }
        assert_eq!(fetched.len() as u64, rows as u64 * shape.fetched_row_bytes());
        let fetched_b = gpu.bytes(&fetched);
        let emb_b = gpu.empty((rows * n * 4) as u64);
        unsafe {
            gpu.q4
                .ngram_decode(&gpu.stream, shape, fetched_b.ptr(), emb_b.ptr(), rows as u32)
                .unwrap();
        }
        report("NgramTable::decode_row", rel_max(&gpu.read(&emb_b, rows * n), &want_emb));

        let key = rng.floats(rows * wide, 2.0);
        let value = rng.floats(rows * n, 1.5);
        let history = rng.floats(ngram.history() * wide, 1.0);
        let (mut want_gate, mut want_gated, mut want_nrm) =
            (vec![0.0f32; rows * hc], vec![0.0f32; rows * wide], vec![0.0f32; rows * wide]);
        for t in 0..rows {
            let row = t * wide..(t + 1) * wide;
            let gate = &mut want_gate[t * hc..(t + 1) * hc];
            pieces::ngram_gate(
                &key[row.clone()],
                &r[row.clone()],
                &nw.norm_key,
                &nw.norm_query,
                c.rms_eps,
                gate,
            );
            pieces::ngram_gated(&value[t * n..(t + 1) * n], gate, &mut want_gated[row.clone()]);
            pieces::stream_norm(
                &want_gated[row.clone()],
                &nw.norm_conv,
                n,
                c.rms_eps,
                &mut want_nrm[row],
            );
        }
        let conv = pieces::conv_dilated(
            &nw.conv,
            ngram.conv_kernel,
            ngram.ngram_size,
            &history,
            &want_nrm,
            wide,
            rows,
        );
        let mut want_r = r.clone();
        pieces::ngram_update(&mut want_r, &want_gated, &conv);

        let r_b = gpu.floats(&r);
        let bufs = [&key, &value, &nw.norm_key, &nw.norm_query, &nw.norm_conv, &nw.conv]
            .map(|x| gpu.floats(x));
        let (gated_b, nrm_b, gate_b) = (
            gpu.empty((rows * wide * 4) as u64),
            gpu.empty((rows * wide * 4) as u64),
            gpu.empty((rows * hc * 4) as u64),
        );
        let history_b = gpu.floats(&history);
        let segments = [Segment { row0: 0, n_rows: rows as u32, slot: 0, pos0: 20, direct: false }];
        let record = StepRecord::new(&segments, &tokens, None, &[]).unwrap();
        let step_b = gpu.bytes(record.as_bytes());
        unsafe {
            gpu.q4
                .ngram_gate(
                    &gpu.stream,
                    shape,
                    c.rms_eps,
                    bufs[0].ptr(),
                    bufs[1].ptr(),
                    r_b.ptr(),
                    bufs[2].ptr(),
                    bufs[3].ptr(),
                    bufs[4].ptr(),
                    gated_b.ptr(),
                    nrm_b.ptr(),
                    Some(gate_b.ptr()),
                    rows as u32,
                )
                .unwrap();
            gpu.q4
                .ngram_window(
                    &gpu.stream,
                    step_b.ptr(),
                    record.dims(),
                    shape,
                    r_b.ptr(),
                    gated_b.ptr(),
                    nrm_b.ptr(),
                    bufs[5].ptr(),
                    GgmlType::F32,
                    history_b.ptr(),
                    shape.history_len(),
                )
                .unwrap();
        }
        report("ngram_gate", rel_max(&gpu.read(&gate_b, rows * hc), &want_gate));
        report("ngram_gated", rel_max(&gpu.read(&gated_b, rows * wide), &want_gated));
        report(
            "stream_norm of the gated value",
            rel_max(&gpu.read(&nrm_b, rows * wide), &want_nrm),
        );
        report("conv_dilated + ngram_update", rel_max(&gpu.read(&r_b, rows * wide), &want_r));
        assert_eq!(gpu.read(&history_b, history.len()), history, "the window wrote the history");

        // 3.6: queries, pooled keys, block scores, selection. One sequence of 61 cells whose
        // last 6 are the rows of the step (so a block that began before the step is completed).
        let ix = indexer_shape(&c);
        let aw = match &w.layers[3].mixer {
            Mixer::Attention(aw) => aw,
            Mixer::Recurrent(_) => unreachable!("layer 3 is an attention layer"),
        };
        let (id, ih, block) = (c.indexer.key_dim, c.indexer.n_head, c.indexer.block);
        let (cells, c0) = (61usize, 55usize);
        let raw_all = rng.floats(cells * id, 2.0);
        let iq = rng.floats(rows * ih * id, 2.0);
        let table_of = |cell: usize| pieces::rope_table(&c.attention.rope, [cell as f64; 3]);
        let mut want_u = vec![0.0f32; rows * ih * id];
        for t in 0..rows {
            for e in 0..ih {
                let at = (t * ih + e) * id;
                pieces::norm_rope(
                    &iq[at..at + id],
                    &aw.indexer_q_norm,
                    c.rms_eps,
                    &table_of(c0 + t),
                    &mut want_u[at..at + id],
                );
            }
        }
        let n_blocks = cells / block;
        let mut want_pooled = vec![0.0f32; n_blocks * id];
        for j in 0..n_blocks {
            pieces::pool_key(
                &raw_all[j * block * id..(j + 1) * block * id],
                &aw.indexer_k_norm,
                c.rms_eps,
                &table_of(j * block),
                &mut want_pooled[j * id..(j + 1) * id],
            );
        }
        // The device state before the step: the keys of the blocks complete at cell 55 in the
        // pool, and the raw keys of cells 52..54 with the rotation of cell 52 carried.
        let known = c0 / block;
        let slot_bytes = ix.key_slot_bytes(KeyFormat::F32);
        let pool_b = gpu.zeros(slot_bytes);
        pool_b
            .write(
                0,
                &want_pooled[..known * id]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
        let mut carried = raw_all[known * block * id..c0 * id].to_vec();
        carried.resize((block - 1) * id, 0.0);
        carried.extend(
            table_of(known * block).iter().flat_map(|&(cos, sin)| [cos as f32, sin as f32]),
        );
        let carry_b = gpu.floats(&carried);
        let attn_rows: Vec<AttnRow> = (0..rows)
            .map(|t| AttnRow { table: 0, pos: (c0 + t) as u32, slot_base: 0, n_slots: 0 })
            .collect();
        let segments =
            [Segment { row0: 0, n_rows: rows as u32, slot: 0, pos0: c0 as i32, direct: false }];
        let record = StepRecord::new(&segments, &tokens, None, &[]).unwrap();
        let (step_b, rows_b) =
            (gpu.bytes(record.as_bytes()), gpu.bytes(&AttnRow::to_bytes(&attn_rows)));
        let rope_b = gpu.empty((rows * c.attention.rope.dim / 2 * 8) as u64);
        let (iq_b, raw_b) = (gpu.floats(&iq), gpu.floats(&raw_all[c0 * id..]));
        let (qn_b, kn_b) = (gpu.floats(&aw.indexer_q_norm), gpu.floats(&aw.indexer_k_norm));
        let key_table = gpu.words(&[0]);
        let stride = n_blocks as u32;
        let scores_b = gpu.empty((rows * n_blocks * 4) as u64);
        unsafe {
            gpu.layers
                .rope_table(&gpu.stream, step_b.ptr(), record.dims(), rope_b.ptr(), rope_params(&c))
                .unwrap();
            gpu.q4
                .indexer_window(
                    &gpu.stream,
                    step_b.ptr(),
                    record.dims(),
                    ix,
                    KeyFormat::F32,
                    c.rms_eps,
                    rows_b.ptr(),
                    iq_b.ptr(),
                    raw_b.ptr(),
                    qn_b.ptr(),
                    kn_b.ptr(),
                    rope_b.ptr(),
                    iq_b.ptr(),
                    carry_b.ptr(),
                    ix.carry_len(),
                    key_table.ptr(),
                    pool_b.ptr(),
                )
                .unwrap();
            gpu.q4
                .block_scores(
                    &gpu.stream,
                    ix,
                    KeyFormat::F32,
                    rows_b.ptr(),
                    iq_b.ptr(),
                    key_table.ptr(),
                    pool_b.ptr(),
                    scores_b.ptr(),
                    0,
                    rows as u32,
                    stride,
                )
                .unwrap();
        }
        report("norm_rope (indexer queries)", rel_max(&gpu.read(&iq_b, rows * ih * id), &want_u));
        report("pool_key", rel_max(&gpu.read(&pool_b, n_blocks * id), &want_pooled));
        let scores = gpu.read(&scores_b, rows * n_blocks);
        let mut worst = 0.0f64;
        for t in 0..rows {
            let visible = (c0 + t + 1) / block;
            let want: Vec<f32> = (0..visible)
                .map(|j| {
                    pieces::block_score(
                        &want_u[t * ih * id..(t + 1) * ih * id],
                        &want_pooled[j * id..(j + 1) * id],
                    )
                })
                .collect();
            worst = worst.max(rel_max(&scores[t * n_blocks..][..visible], &want));
        }
        report("block_score", worst);
        for (rule, device_rule) in [
            (SelectionRule::CellBudget, DeviceRule::CellBudget),
            (SelectionRule::WholeBlocks, DeviceRule::WholeBlocks),
        ] {
            let cap = ix.max_ranges(0);
            let lists = gpu.empty((rows as u32 * cap * 12).into());
            let counts = gpu.empty(rows as u64 * 16);
            unsafe {
                gpu.q4
                    .select(
                        &gpu.stream,
                        ix,
                        device_rule,
                        rows_b.ptr(),
                        scores_b.ptr(),
                        None,
                        lists.ptr(),
                        counts.ptr(),
                        0,
                        rows as u32,
                        stride,
                        cap,
                    )
                    .unwrap();
            }
            let (lists, counts) = (
                gpu.read_words(&lists, 0, rows * cap as usize * 3),
                gpu.read_words(&counts, 0, rows * 4),
            );
            for t in 0..rows {
                let visible = c0 + t + 1;
                let want = pieces::select_cells(
                    rule,
                    &c.indexer,
                    visible,
                    &scores[t * n_blocks..][..visible / block],
                );
                let list = &lists[t * cap as usize * 3..][..counts[t * 4] as usize * 3];
                let got: Vec<u32> =
                    list.as_chunks::<3>().0.iter().flat_map(|r| r[0]..r[1]).collect();
                assert_eq!(got, want, "select_cells, {rule:?}, row {t}");
                assert_eq!(counts[t * 4 + 1] as usize, want.len());
            }
        }
        eprintln!("    select_cells: equal under both rules");
    }
}

// ---------------------------------------------------------------------------
// The chain.

/// What an attention layer keeps on the device.
struct AttentionLayer {
    q: DeviceMatrix,
    k: DeviceMatrix,
    v: DeviceMatrix,
    out: DeviceMatrix,
    indexer_q: DeviceMatrix,
    indexer_k: DeviceMatrix,
    norms: [DeviceBuffer; 4],
    exact_pool: DeviceBuffer,
    conv_pool: DeviceBuffer,
    key_pool: DeviceBuffer,
    carry: DeviceBuffer,
}

fn attention_layer(
    gpu: &Gpu,
    c: &Config,
    w: &AttentionWeights<'_>,
    geo: &CacheGeometry,
    entries: u64,
) -> AttentionLayer {
    let ix = indexer_shape(c);
    AttentionLayer {
        q: device_matrix(gpu, &w.q),
        k: device_matrix(gpu, &w.k),
        v: device_matrix(gpu, &w.v),
        out: device_matrix(gpu, &w.out),
        indexer_q: device_matrix(gpu, &w.indexer_q),
        indexer_k: device_matrix(gpu, &w.indexer_k),
        norms: [&w.q_norm, &w.k_norm, &w.indexer_q_norm, &w.indexer_k_norm].map(|x| gpu.floats(x)),
        exact_pool: gpu.empty(geo.exact_entry_bytes()),
        conv_pool: gpu.empty(geo.converted_entry_bytes() * entries),
        key_pool: gpu.empty(ix.key_slot_bytes(KeyFormat::F32) * entries),
        carry: gpu.empty(ix.carry_len() * 4),
    }
}

/// What a step shares between its attention layers.
struct StepAttention {
    step: DeviceBuffer,
    dims: StepDims,
    rope: DeviceBuffer,
    plan: Plan,
    plan_b: DeviceBuffer,
    hands: DeviceBuffer,
    cap: u32,
}

const PIECES: u32 = 2;

fn step_attention(
    gpu: &Gpu,
    c: &Config,
    geo: &CacheGeometry,
    record: &StepRecord,
    c0: usize,
) -> StepAttention {
    let rows = record.dims().n_rows;
    let cap = indexer_shape(c).max_ranges(0);
    // Every row's list is made on the device: two pieces of converted entries per row.
    let mut plan = Plan {
        rows: (0..rows)
            .map(|t| AttnRow {
                table: 0,
                pos: c0 as u32 + t,
                slot_base: t * PIECES,
                n_slots: PIECES,
            })
            .collect(),
        row_ids: (0..rows).collect(),
        ranges: vec![0; (rows * cap * 3) as usize],
        state_slots: rows * PIECES,
        needs_merge: true,
        ..Plan::default()
    };
    for row in 0..rows {
        for piece in 0..PIECES {
            for kv_head in 0..geo.kv_heads {
                // Flag 1: the item reads converted entries.
                plan.items.push(AttnItem {
                    row0: row,
                    n_rows: 1,
                    kv_head,
                    flags: 1,
                    slot: piece,
                    ..AttnItem::default()
                });
            }
        }
    }
    plan.groups.push((AttnShape::for_vecs(geo.q_per_kv), 0..plan.items.len()));
    let hands: Vec<ListHandOver> = (0..rows)
        .map(|row| ListHandOver {
            ranges_off: row * cap * 3,
            item_conv: row * PIECES * geo.kv_heads,
            pieces_conv: PIECES,
            kv_heads: geo.kv_heads,
            ..ListHandOver::default()
        })
        .collect();
    let mut bytes = vec![0u8; plan.layout().bytes as usize];
    plan.write(&mut bytes);
    let step = gpu.bytes(record.as_bytes());
    let rope = gpu.empty(u64::from(rows) * (c.attention.rope.dim as u64 / 2) * 8);
    unsafe {
        gpu.layers
            .rope_table(&gpu.stream, step.ptr(), record.dims(), rope.ptr(), rope_params(c))
            .unwrap();
    }
    StepAttention {
        step,
        dims: record.dims(),
        rope,
        plan_b: gpu.bytes(&bytes),
        plan,
        hands: gpu.bytes(&ListHandOver::to_bytes(&hands)),
        cap,
    }
}

/// What a layer's window keeps for the commit of a segment with drafts.
struct Kept {
    raw: DeviceBuffer,
}

/// An attention layer on the device: from the mixed input `m` to the layer's output, with the
/// cells every row selected.
#[allow(clippy::too_many_arguments)]
fn run_attention(
    gpu: &Gpu,
    c: &Config,
    geo: &CacheGeometry,
    layer: &AttentionLayer,
    sa: &StepAttention,
    tables: &DeviceBuffer,
    key_table: &DeviceBuffer,
    m: &Act,
    direct: bool,
) -> (DeviceBuffer, Vec<Vec<u32>>, Kept) {
    let rows = sa.dims.n_rows as usize;
    let (a, ix) = (&c.attention, indexer_shape(c));
    let (q_dim, kv_dim) = c.attention_dims();
    let layout = sa.plan.layout();
    let qg = product(gpu, &layer.q, m, rows);
    let k_in = product(gpu, &layer.k, m, rows);
    let v = product(gpu, &layer.v, m, rows);
    let iq = product(gpu, &layer.indexer_q, m, rows);
    let raw = product(gpu, &layer.indexer_k, m, rows);
    let q = gpu.empty((rows * q_dim * 4) as u64);
    let gate = gpu.empty((rows * q_dim * 4) as u64);
    let k = gpu.empty((rows * kv_dim * 4) as u64);
    let status = gpu.words(&[0]);
    let n_blocks = (sa.plan.rows[rows - 1].pos + 1) / ix.block;
    let scores = gpu.empty((rows as u64 * u64::from(n_blocks.max(1))) * 4);
    let counts = gpu.empty(rows as u64 * 16);
    let heads = Act::new(gpu, q_dim, rows);
    let scratch = gpu.empty(sa.plan.state_bytes(geo));
    let (plan_rows, ranges, items) = (
        sa.plan_b.ptr().offset(layout.rows),
        sa.plan_b.ptr().offset(layout.ranges),
        sa.plan_b.ptr().offset(layout.items),
    );
    unsafe {
        gpu.layers
            .attn_prep(
                &gpu.stream,
                sa.step.ptr(),
                sa.dims,
                AttentionShape {
                    n_head: a.n_head as u32,
                    n_kv_head: a.n_head_kv as u32,
                    head_dim: a.head_dim as u32,
                },
                a.rope.dim as u32,
                c.rms_eps,
                qg.ptr(),
                k_in.ptr(),
                layer.norms[0].ptr(),
                layer.norms[1].ptr(),
                sa.rope.ptr(),
                q.ptr(),
                Some(gate.ptr()),
                k.ptr(),
            )
            .unwrap();
        gpu.attention
            .append(
                &gpu.stream,
                geo,
                plan_rows,
                rows as u32,
                k.ptr(),
                v.ptr(),
                tables.ptr(),
                layer.exact_pool.ptr(),
                layer.conv_pool.ptr(),
                status.ptr(),
            )
            .unwrap();
        gpu.q4
            .indexer_window(
                &gpu.stream,
                sa.step.ptr(),
                sa.dims,
                ix,
                KeyFormat::F32,
                c.rms_eps,
                plan_rows,
                iq.ptr(),
                raw.ptr(),
                layer.norms[2].ptr(),
                layer.norms[3].ptr(),
                sa.rope.ptr(),
                iq.ptr(),
                layer.carry.ptr(),
                ix.carry_len(),
                key_table.ptr(),
                layer.key_pool.ptr(),
            )
            .unwrap();
        if direct {
            gpu.q4
                .indexer_commit(
                    &gpu.stream,
                    sa.step.ptr(),
                    sa.dims,
                    ix,
                    CommitSet::Direct,
                    plan_rows,
                    raw.ptr(),
                    sa.rope.ptr(),
                    layer.carry.ptr(),
                    ix.carry_len(),
                )
                .unwrap();
        }
        gpu.q4
            .block_scores(
                &gpu.stream,
                ix,
                KeyFormat::F32,
                plan_rows,
                iq.ptr(),
                key_table.ptr(),
                layer.key_pool.ptr(),
                scores.ptr(),
                0,
                rows as u32,
                n_blocks.max(1),
            )
            .unwrap();
        gpu.q4
            .select(
                &gpu.stream,
                ix,
                DeviceRule::CellBudget,
                plan_rows,
                scores.ptr(),
                Some(tables.ptr()),
                ranges,
                counts.ptr(),
                0,
                rows as u32,
                n_blocks.max(1),
                sa.cap,
            )
            .unwrap();
    }
    // The lists as the select kernel left them, before the hand-over rebases them.
    let lists = gpu.read_words(&sa.plan_b, layout.ranges, rows * sa.cap as usize * 3);
    let counts_host = gpu.read_words(&counts, 0, rows * 4);
    let selected: Vec<Vec<u32>> = (0..rows)
        .map(|t| {
            assert_eq!(counts_host[t * 4 + 2], 0, "all entries are converted");
            lists[t * sa.cap as usize * 3..][..counts_host[t * 4] as usize * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|r| r[0]..r[1])
                .collect()
        })
        .collect();
    let buffers = unsafe { sa.plan.on_device(sa.plan_b.ptr()) };
    unsafe {
        gpu.q4
            .hand_over(&gpu.stream, sa.hands.ptr(), counts.ptr(), ranges, items, rows as u32)
            .unwrap();
        gpu.attention
            .attend(
                &gpu.stream,
                geo,
                &buffers,
                &AttendBuffers {
                    tables: tables.ptr(),
                    exact_pool: layer.exact_pool.ptr(),
                    converted_pool: layer.conv_pool.ptr(),
                    q: q.ptr(),
                    gate: Some(gate.ptr()),
                    out: heads.values(),
                    scratch: scratch.ptr(),
                },
                1.0 / (a.head_dim as f32).sqrt(),
            )
            .unwrap();
    }
    let out = product(gpu, &layer.out, &heads, rows);
    assert_eq!(gpu.read_words(&status, 0, 1), [0], "the K/V append reported a problem");
    (out, selected, Kept { raw })
}

#[test]
fn a_chain_of_steps_matches_the_reference_layer_by_layer() {
    let Some(gpus) = gpus() else { return };
    let synthetic = model();
    let gguf = open("chain", &synthetic.bytes());
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let (hc, n, wide) = (c.n_stream, c.n_embd, c.wide());
    let ngram = c.ngram.as_ref().unwrap();
    let ng_shape = ngram_shape(&c);
    let table = w.ngram_table.as_ref().unwrap();
    let a = &c.attention;
    let geo = CacheGeometry {
        k: CacheFormat::F16,
        v: CacheFormat::F16,
        head_dim: a.head_dim as u32,
        kv_heads: a.n_head_kv as u32,
        kv_total: a.n_head_kv as u32,
        first_kv_head: 0,
        q_per_kv: (a.n_head / a.n_head_kv) as u32,
    };
    // Windows of up to 8 rows: prompt rows and rows without drafts are direct, windows with
    // drafts are committed at the end with the accepted count.
    let steps: [(usize, bool, usize); 8] = [
        (8, true, 8),
        (5, false, 2),
        (1, true, 1),
        (6, false, 6),
        (8, true, 8),
        (4, false, 0),
        (7, false, 3),
        (8, true, 8),
    ];
    let all_tokens = synthetic.shape.tokens(steps.iter().map(|s| s.0).sum(), 3);
    for device in gpus {
        let gpu = Gpu::new(device);
        eprintln!("  {}", device.info().name);
        // The model on the device.
        let mixers: Vec<[DeviceMixer; 2]> = w
            .layers
            .iter()
            .map(|l| [device_mixer(&gpu, &l.attn_mixer), device_mixer(&gpu, &l.ffn_mixer)])
            .collect();
        let entries = 2u64;
        let attention: Vec<Option<AttentionLayer>> = w
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Attention(aw) => Some(attention_layer(&gpu, &c, aw, &geo, entries)),
                Mixer::Recurrent(_) => None,
            })
            .collect();
        let nw = w.layers[ngram.layer].ngram.as_ref().unwrap();
        let (ple_key, ple_value) = (device_matrix(&gpu, &nw.key), device_matrix(&gpu, &nw.value));
        let ple = [&nw.norm_key, &nw.norm_query, &nw.norm_conv, &nw.conv].map(|x| gpu.floats(x));
        let history = gpu.zeros(ng_shape.history_len() * 4);
        // Both entries of the sequence are converted entries (f16 rows are written straight
        // into them); the key slots are swapped against the entries.
        let tables = gpu.words(&[1 << 31, 1 << 31 | 1]);
        let key_table = gpu.words(&[1, 0]);

        let mut state = SequenceState::new(&c);
        let mut worst = std::collections::BTreeMap::<&'static str, f64>::new();
        let mut note = |name: &'static str, got: &[f32], want: &[f32]| {
            let e = rel_max(got, want);
            let slot = worst.entry(name).or_insert(0.0);
            *slot = slot.max(e);
        };
        let (mut rows_selecting, mut rows_same_cells) = (0usize, 0usize);
        let mut at = 0usize;
        for &(rows, direct, accepted) in &steps {
            let tokens = &all_tokens[at..at + rows];
            let c0 = state.position();
            let segments =
                [Segment { row0: 0, n_rows: rows as u32, slot: 0, pos0: c0 as i32, direct }];
            let mut record = StepRecord::new(&segments, tokens, None, &[]).unwrap();
            let sa = step_attention(&gpu, &c, &geo, &record, c0);

            // 3.1 on the host: the embedding in every stream.
            let mut r0 = vec![0.0f32; rows * wide];
            for (row, &token) in tokens.iter().enumerate() {
                for stream in r0[row * wide..(row + 1) * wide].chunks_exact_mut(n) {
                    w.token_embd.decode_row(token as usize, stream);
                }
            }
            let r_b = gpu.floats(&r0);
            let mut kept: Vec<Option<Kept>> = Vec::new();
            let mut nrm_kept = None;
            for layer in 0..c.n_layer {
                let r_in = gpu.read(&r_b, rows * wide);
                let mut rec = Recorder::default();
                let want_out = reference.block_observed(layer, &state, tokens, &r_in, &mut rec);

                if layer == ngram.layer {
                    // The host's part: the hash and the rows of the table.
                    let mut before = state.recent_tokens().to_vec();
                    let mut fetched = Vec::new();
                    for &token in tokens {
                        for id in pieces::ngram_rows(ngram, token, &before) {
                            fetched.extend_from_slice(table.row_bytes(id));
                        }
                        before.push(token);
                    }
                    let fetched_b = gpu.bytes(&fetched);
                    let emb = Act::new(&gpu, n, rows);
                    unsafe {
                        gpu.q4
                            .ngram_decode(
                                &gpu.stream,
                                ng_shape,
                                fetched_b.ptr(),
                                emb.values(),
                                rows as u32,
                            )
                            .unwrap();
                    }
                    let key = product(&gpu, &ple_key, &emb, rows);
                    let value = product(&gpu, &ple_value, &emb, rows);
                    let gated = gpu.empty((rows * wide * 4) as u64);
                    let nrm = gpu.empty((rows * wide * 4) as u64);
                    unsafe {
                        gpu.q4
                            .ngram_gate(
                                &gpu.stream,
                                ng_shape,
                                c.rms_eps,
                                key.ptr(),
                                value.ptr(),
                                r_b.ptr(),
                                ple[0].ptr(),
                                ple[1].ptr(),
                                ple[2].ptr(),
                                gated.ptr(),
                                nrm.ptr(),
                                None,
                                rows as u32,
                            )
                            .unwrap();
                        gpu.q4
                            .ngram_window(
                                &gpu.stream,
                                sa.step.ptr(),
                                sa.dims,
                                ng_shape,
                                r_b.ptr(),
                                gated.ptr(),
                                nrm.ptr(),
                                ple[3].ptr(),
                                GgmlType::F32,
                                history.ptr(),
                                ng_shape.history_len(),
                            )
                            .unwrap();
                        gpu.q4
                            .ngram_commit(
                                &gpu.stream,
                                sa.step.ptr(),
                                sa.dims,
                                ng_shape,
                                CommitSet::Direct,
                                nrm.ptr(),
                                history.ptr(),
                                ng_shape.history_len(),
                            )
                            .unwrap();
                    }
                    note(
                        "ngram_embedding",
                        &gpu.read_at(&emb.buf, emb.f32_at, rows * n),
                        rec.tensor_of(Some(layer), "ngram_embedding").unwrap(),
                    );
                    note(
                        "ngram_out",
                        &gpu.read(&r_b, rows * wide),
                        rec.tensor_of(Some(layer), "ngram_out").unwrap(),
                    );
                    nrm_kept = Some(nrm);
                }

                let (m, inj) = read(&gpu, &c, &mixers[layer][0], &r_b, rows, None);
                note(
                    "attn_in",
                    &gpu.read_at(&m.buf, m.f32_at, rows * n),
                    rec.tensor_of(Some(layer), "attn_in").unwrap(),
                );
                let want_attn = rec.tensor_of(Some(layer), "attn_out").unwrap();
                let attn_out = match &attention[layer] {
                    Some(al) => {
                        let (out, selected, keep) =
                            run_attention(&gpu, &c, &geo, al, &sa, &tables, &key_table, &m, direct);
                        let got = gpu.read(&out, rows * n);
                        for (row, cells) in selected.iter().enumerate() {
                            let (want_cells, _) = &rec.selection[&(layer, row)];
                            rows_selecting += usize::from(c0 + row + 1 > c.indexer.dense_limit());
                            if cells == want_cells {
                                rows_same_cells += 1;
                                note(
                                    "attn_out (attention)",
                                    &got[row * n..(row + 1) * n],
                                    &want_attn[row * n..(row + 1) * n],
                                );
                            } else {
                                eprintln!(
                                    "    layer {layer} row {row}: other cells than the reference"
                                );
                            }
                        }
                        kept.push(Some(keep));
                        out
                    }
                    None => {
                        kept.push(None);
                        // The recurrent part is LayerKernels' and is not run here.
                        gpu.floats(want_attn)
                    }
                };
                let (m, inj) =
                    read(&gpu, &c, &mixers[layer][1], &r_b, rows, Some((&attn_out, &inj)));
                note(
                    if attention[layer].is_some() { "ffn_in (attention layers)" } else { "ffn_in" },
                    &gpu.read_at(&m.buf, m.f32_at, rows * n),
                    rec.tensor_of(Some(layer), "ffn_in").unwrap(),
                );
                // The mixture of experts is ExpertKernels' and is not run here.
                let ffn_out = gpu.floats(rec.tensor_of(Some(layer), "ffn_out").unwrap());
                unsafe {
                    gpu.q4
                        .stream_write(
                            &gpu.stream,
                            r_b.ptr(),
                            ffn_out.ptr(),
                            inj.ptr(),
                            rows as u32,
                            hc as u32,
                            n as u32,
                        )
                        .unwrap();
                }
                note(
                    if attention[layer].is_some() {
                        "layer_out (attention layers)"
                    } else {
                        "layer_out"
                    },
                    &gpu.read(&r_b, rows * wide),
                    &want_out,
                );
            }

            // The end of the step: the verdict, the deferred commits, and the reference's own.
            if !direct {
                record.set_accepted(0, accepted as u32).unwrap();
                let step_b = gpu.bytes(record.as_bytes());
                let plan_rows = sa.plan_b.ptr().offset(sa.plan.layout().rows);
                unsafe {
                    gpu.q4
                        .ngram_commit(
                            &gpu.stream,
                            step_b.ptr(),
                            sa.dims,
                            ng_shape,
                            CommitSet::Deferred,
                            nrm_kept.as_ref().unwrap().ptr(),
                            history.ptr(),
                            ng_shape.history_len(),
                        )
                        .unwrap();
                    for (al, keep) in attention.iter().zip(&kept) {
                        if let (Some(al), Some(keep)) = (al, keep) {
                            gpu.q4
                                .indexer_commit(
                                    &gpu.stream,
                                    step_b.ptr(),
                                    sa.dims,
                                    indexer_shape(&c),
                                    CommitSet::Deferred,
                                    plan_rows,
                                    keep.raw.ptr(),
                                    sa.rope.ptr(),
                                    al.carry.ptr(),
                                    indexer_shape(&c).carry_len(),
                                )
                                .unwrap();
                        }
                    }
                }
                gpu.stream.synchronize().unwrap();
            }
            let window = reference.window(&state, tokens, LogitRows::None).unwrap();
            reference.commit(&mut state, &window, accepted).unwrap();
            at += accepted;

            // The device's n-gram history is the reference's.
            note(
                "ngram history",
                &gpu.read(&history, state.ngram_history().len()),
                state.ngram_history(),
            );
        }
        for (name, value) in &worst {
            report(&format!("chain {name} rel_max"), *value);
        }
        eprintln!(
            "    attention rows: {rows_same_cells} of {} with the reference's cells, {rows_selecting} past the dense limit",
            2 * steps.iter().map(|s| s.0).sum::<usize>()
        );
        for name in
            ["ngram_embedding", "ngram_out", "attn_in", "ffn_in", "layer_out", "ngram history"]
        {
            assert!(worst[name] < 2e-5, "{name}: {}", worst[name]);
        }
        // The stored K and V are f16 and the products run on f16 tensor cores; what follows the
        // attention in its layer carries that error on.
        for name in
            ["attn_out (attention)", "ffn_in (attention layers)", "layer_out (attention layers)"]
        {
            assert!(worst[name] < 6e-3, "{name}: {}", worst[name]);
        }
        assert!(rows_same_cells * 10 >= 9 * 2 * steps.iter().map(|s| s.0).sum::<usize>());
    }
}

// ---------------------------------------------------------------------------
// The lists of a real model (CPU only).

/// An observer that keeps the selected cells of the rows past the dense limit.
struct Lists {
    first_cell: usize,
    limit: usize,
    by_layer: std::collections::BTreeMap<usize, Vec<Vec<u32>>>,
}

impl ares_model::qwen4exp::Observer for Lists {
    fn selection(&mut self, layer: usize, row: usize, cells: &[u32], _scores: &[f32]) {
        if self.first_cell + row >= self.limit {
            self.by_layer.entry(layer).or_default().push(cells.to_vec());
        }
    }
}

/// How much the lists of neighbouring rows of a real model overlap, which decides whether
/// rows of a prompt piece can share attention items. Needs no GPU:
///
/// ```sh
/// ARES_TEST_QWEN4EXP_MODEL=<first part of the model> ARES_TEST_QWEN4EXP_TOKENS=<file of token ids> \
///   cargo test -p ares-model --test qwen4exp_gpu neighbouring -- --nocapture
/// ```
///
/// `ARES_TEST_QWEN4EXP_LAYERS` (default 8) is the number of layers that are run: the lists
/// of the attention layers among them are measured.
#[test]
fn neighbouring_rows_of_a_real_model_select_overlapping_cells() {
    let (Some(model), Some(tokens)) = (
        std::env::var_os("ARES_TEST_QWEN4EXP_MODEL"),
        std::env::var_os("ARES_TEST_QWEN4EXP_TOKENS"),
    ) else {
        eprintln!("skipped: set ARES_TEST_QWEN4EXP_MODEL and ARES_TEST_QWEN4EXP_TOKENS");
        return;
    };
    let depth =
        std::env::var("ARES_TEST_QWEN4EXP_LAYERS").ok().and_then(|d| d.parse().ok()).unwrap_or(8);
    let gguf = Gguf::open(model).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let tokens: Vec<u32> = std::fs::read_to_string(tokens)
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().expect("a token id"))
        .collect();
    let reference = Reference::new(&c, &w).first_layers(depth).keep_residual(false);
    let limit = c.indexer.dense_limit();
    let mut state = SequenceState::new(&c);
    let mut lists = Lists { first_cell: 0, limit, by_layer: Default::default() };
    let started = std::time::Instant::now();
    while state.position() < tokens.len() {
        let at = state.position();
        let rows = if at < limit { 128.min(limit - at) } else { 32.min(tokens.len() - at) };
        lists.first_cell = at;
        reference
            .forward_observed(&mut state, &tokens[at..at + rows], LogitRows::None, &mut lists)
            .unwrap();
        eprintln!("    {} cells after {:.0} s", at + rows, started.elapsed().as_secs_f64());
    }
    let shared = |a: &[u32], b: &[u32]| a.iter().filter(|c| b.binary_search(c).is_ok()).count();
    for (layer, rows) in &lists.by_layer {
        eprintln!(
            "  layer {layer}: {} rows past the dense limit, {} cells each",
            rows.len(),
            rows[0].len()
        );
        for distance in [1usize, 2, 4, 8, 32] {
            let overlaps: Vec<f64> = rows
                .iter()
                .zip(rows.iter().skip(distance))
                .map(|(a, b)| shared(a, b) as f64 / a.len() as f64)
                .collect();
            if overlaps.is_empty() {
                continue;
            }
            eprintln!(
                "    rows {distance} apart share {:.1} % of their cells on average, {:.1} % at least",
                100.0 * overlaps.iter().sum::<f64>() / overlaps.len() as f64,
                100.0 * overlaps.iter().copied().fold(1.0, f64::min)
            );
        }
        for group in [8usize, 32] {
            let unions: Vec<f64> = rows
                .chunks_exact(group)
                .map(|g| {
                    let mut all: Vec<u32> = g.iter().flatten().copied().collect();
                    all.sort_unstable();
                    all.dedup();
                    let common = g[0]
                        .iter()
                        .filter(|c| g.iter().all(|r| r.binary_search(c).is_ok()))
                        .count();
                    eprintln!(
                        "      {group} rows: {} cells in common, {} in all",
                        common,
                        all.len()
                    );
                    all.len() as f64 / g[0].len() as f64
                })
                .collect();
            eprintln!(
                "    the union of {group} consecutive rows is {:.2} lists on average",
                unions.iter().sum::<f64>() / unions.len().max(1) as f64
            );
        }
        let ranges: Vec<usize> =
            rows.iter().map(|r| 1 + r.windows(2).filter(|w| w[1] != w[0] + 1).count()).collect();
        eprintln!(
            "    ranges per row: {:.0} on average, {} to {}",
            ranges.iter().sum::<usize>() as f64 / ranges.len() as f64,
            ranges.iter().min().unwrap(),
            ranges.iter().max().unwrap()
        );
    }
}
