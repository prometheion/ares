//! Tests of the `qwen4exp` kernels (`kernels/q4.cu`, `src/q4.rs`) on random
//! inputs against plain f64 formulas written here; the same kernels are held to
//! `ares_model`'s reference in `crates/model/tests/qwen4exp_gpu.rs`.
//!
//! Like the other GPU tests they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test q4 -- --test-threads=1 --nocapture
//! ```

// The tests launch kernels on buffers they allocated a few lines earlier, with sizes taken from
// the same shapes; a comment on each of the launches would repeat that.
#![allow(clippy::undocumented_unsafe_blocks)]

use ares_core::GgmlType;
use ares_cuda::{
    AttendBuffers, AttentionKernels, AttnItem, AttnRow, AttnShape, CacheFormat, CacheGeometry,
    CommitSet, Device, DeviceBuffer, IndexerShape, Kernels, KeyFormat, LayerKernels, ListHandOver,
    MixerMatrix, NgramShape, Plan, Q4Kernels, RopeParams, Segment, SelectionRule, StepRecord,
    Stream, quantized_activation_bytes,
};
use common::blocks::{TYPES, random_blocks};
use common::{Rng, gpus, rel_max};

mod common;

struct Gpu {
    device: Device,
    stream: Stream,
    q4: Q4Kernels,
    layers: LayerKernels,
    kernels: Kernels,
}

impl Gpu {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            stream: Stream::new(device, 0).unwrap(),
            q4: Q4Kernels::load(device).unwrap(),
            layers: LayerKernels::load(device).unwrap(),
            kernels: Kernels::load(device).unwrap(),
        }
    }
    fn name(&self) -> String {
        self.device.info().name.clone()
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
    /// A buffer of `bytes` bytes, poisoned so that a kernel that does not write what it should
    /// is seen.
    fn empty(&self, bytes: u64) -> DeviceBuffer {
        let buffer = DeviceBuffer::new(&self.device, bytes.max(16)).unwrap();
        buffer.write(0, &vec![0xFFu8; bytes as usize]).unwrap();
        buffer
    }
    fn read_bytes(&self, buffer: &DeviceBuffer, offset: u64, n: usize) -> Vec<u8> {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; n];
        buffer.read(offset, &mut bytes).unwrap();
        bytes
    }
    fn read(&self, buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.read_at(buffer, 0, n)
    }
    fn read_at(&self, buffer: &DeviceBuffer, offset: u64, n: usize) -> Vec<f32> {
        let bytes = self.read_bytes(buffer, offset, n * 4);
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }
    fn read_words(&self, buffer: &DeviceBuffer, n: usize) -> Vec<u32> {
        let bytes = self.read_bytes(buffer, 0, n * 4);
        bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect()
    }
}

fn report(name: &str, value: f64) {
    eprintln!("    {name}: {value:.3e}");
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// `x / sqrt(mean(x^2) + eps) * w`.
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f64> {
    let mean = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean + f64::from(eps)).sqrt();
    x.iter().zip(w).map(|(&v, &w)| f64::from(v) * scale * f64::from(w)).collect()
}

fn to_f32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}

const EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// The stream mixer.

#[test]
fn the_stream_mixer_matches_its_formulas() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x51ab_77e1_0235_9d0f);
        let mut worst = [0.0f64; 5];
        for (rows, hc, n, rank) in
            [(1usize, 4usize, 2560usize, 320usize), (5, 3, 96, 5), (40, 4, 2560, 320)]
        {
            let d = hc * n;
            let r = rng.floats(rows * d, 3.0);
            let norm = rng.floats(d, 1.5);
            let out = rng.floats(rows * n, 2.0);
            let inj = rng.floats(rows * hc, 6.0);
            let (rows32, hc32, n32) = (rows as u32, hc as u32, n as u32);

            // The write, in f64.
            let mut written = vec![0.0f64; rows * d];
            for t in 0..rows {
                for c in 0..hc {
                    let factor = 2.0 * sigmoid(f64::from(inj[t * hc + c]) / hc as f64);
                    for i in 0..n {
                        let at = t * d + c * n + i;
                        written[at] = f64::from(r[at]) + f64::from(out[t * n + i]) * factor;
                    }
                }
            }
            let norm_of = |r: &[f32]| -> Vec<f64> {
                let mut xn = Vec::with_capacity(rows * d);
                for t in 0..rows {
                    for c in 0..hc {
                        let at = t * d + c * n;
                        xn.extend(rms_norm(&r[at..at + n], &norm[c * n..(c + 1) * n], EPS));
                    }
                }
                xn
            };

            let (norm_b, out_b, inj_b) = (gpu.floats(&norm), gpu.floats(&out), gpu.floats(&inj));
            // Norm alone.
            let r_b = gpu.floats(&r);
            let xn_b = gpu.empty((rows * d * 4) as u64);
            unsafe {
                gpu.q4
                    .stream_norm(
                        &gpu.stream,
                        r_b.ptr(),
                        norm_b.ptr(),
                        xn_b.ptr(),
                        rows32,
                        hc32,
                        n32,
                        EPS,
                    )
                    .unwrap();
            }
            worst[0] = worst[0].max(rel_max(&gpu.read(&xn_b, rows * d), &norm_of(&r)));
            assert_eq!(gpu.read(&r_b, rows * d), r, "the norm does not write the residual");
            // Write alone.
            unsafe {
                gpu.q4
                    .stream_write(
                        &gpu.stream,
                        r_b.ptr(),
                        out_b.ptr(),
                        inj_b.ptr(),
                        rows32,
                        hc32,
                        n32,
                    )
                    .unwrap();
            }
            let wrote = gpu.read(&r_b, rows * d);
            worst[1] = worst[1].max(rel_max(&wrote, &written));
            // Write and norm in one pass give the same residual and its norm.
            let r_b = gpu.floats(&r);
            let xn_b = gpu.empty((rows * d * 4) as u64);
            unsafe {
                gpu.q4
                    .stream_write_norm(
                        &gpu.stream,
                        r_b.ptr(),
                        out_b.ptr(),
                        inj_b.ptr(),
                        norm_b.ptr(),
                        xn_b.ptr(),
                        rows32,
                        hc32,
                        n32,
                        EPS,
                    )
                    .unwrap();
            }
            assert_eq!(gpu.read(&r_b, rows * d), wrote);
            worst[2] =
                worst[2].max(rel_max(&gpu.read(&xn_b, rows * d), &norm_of(&to_f32(&written))));

            // The squash.
            let down = rng.floats(rows * rank, 8.0);
            let down_b = gpu.floats(&down);
            let lo_b = gpu.empty((rows * rank * 4) as u64);
            unsafe {
                gpu.q4
                    .squash(&gpu.stream, down_b.ptr(), lo_b.ptr(), (rows * rank) as u32, hc32)
                    .unwrap();
            }
            let want: Vec<f64> = down.iter().map(|&x| silu(f64::from(x) / hc as f64)).collect();
            worst[3] = worst[3].max(rel_max(&gpu.read(&lo_b, rows * rank), &want));

            // The gated mean, and its quantization byte for byte.
            if n % 32 == 0 {
                let xn = rng.floats(rows * d, 2.0);
                let gate = rng.floats(rows * d, 5.0);
                let (xn_b, gate_b) = (gpu.floats(&xn), gpu.floats(&gate));
                let bytes = quantized_activation_bytes((rows * n) as u64);
                let act = gpu.empty(bytes);
                unsafe {
                    gpu.q4
                        .mix_quant(
                            &gpu.stream,
                            xn_b.ptr(),
                            gate_b.ptr(),
                            act.ptr(),
                            rows32,
                            hc32,
                            n32,
                        )
                        .unwrap();
                }
                let mut want = vec![0.0f64; rows * n];
                for t in 0..rows {
                    for i in 0..n {
                        let sum: f64 = (0..hc)
                            .map(|c| {
                                let at = t * d + c * n + i;
                                f64::from(xn[at]) * sigmoid(f64::from(gate[at]))
                            })
                            .sum();
                        want[t * n + i] = sum / hc as f64;
                    }
                }
                let m = gpu.read_at(&act, bytes - (rows * n * 4) as u64, rows * n);
                worst[4] = worst[4].max(rel_max(&m, &want));
                // The whole layout is what the quantization kernel writes for these values. It
                // takes up to 8 rows per launch, which is a limit of its interface only: here the
                // values are one row.
                if rows * n <= 8 * 2560 {
                    let m_b = gpu.floats(&m);
                    let again = gpu.empty(bytes);
                    unsafe {
                        gpu.kernels
                            .quantize_activations(
                                &gpu.stream,
                                m_b.ptr(),
                                again.ptr(),
                                (rows * n) as u32,
                                1,
                            )
                            .unwrap();
                    }
                    assert_eq!(
                        gpu.read_bytes(&act, 0, bytes as usize),
                        gpu.read_bytes(&again, 0, bytes as usize),
                        "the fused quantization differs from quantize_activations"
                    );
                }
            }
        }
        eprintln!("  {}", gpu.name());
        for (name, value) in ["norm", "write", "write+norm", "squash", "mix"].iter().zip(worst) {
            report(&format!("stream mixer {name} rel_max"), value);
            assert!(value < 2e-6, "{name}: {value}");
        }
    }
}

/// The f64 reference of a fused read of `rows` rows (no write in front): the mixed input
/// `m` (`rows * n`) and the inject values (`rows * hc`), from the residual `r`, the norm
/// weights and the decoded matrices (`down`: `rank` rows of `hc * n`, `inject`: `hc` rows of
/// `hc * n`, `up`: `hc * n` rows of `rank`).
#[allow(clippy::too_many_arguments)]
fn read_reference(
    r: &[f32],
    norm: &[f32],
    down: &[f32],
    inject: &[f32],
    up: &[f32],
    (rows, hc, n, rank): (usize, usize, usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let d = hc * n;
    let (mut m, mut inj) = (Vec::with_capacity(rows * n), Vec::with_capacity(rows * hc));
    for t in 0..rows {
        let xn: Vec<f64> = (0..hc)
            .flat_map(|c| rms_norm(&r[t * d + c * n..][..n], &norm[c * n..][..n], EPS))
            .collect();
        let dot =
            |row: &[f32]| -> f64 { row.iter().zip(&xn).map(|(w, x)| f64::from(*w) * x).sum() };
        let lo: Vec<f64> = (0..rank).map(|j| silu(dot(&down[j * d..][..d]) / hc as f64)).collect();
        inj.extend((0..hc).map(|k| dot(&inject[k * d..][..d])));
        for i in 0..n {
            let mut sum = 0.0;
            for c in 0..hc {
                let row = &up[(c * n + i) * rank..][..rank];
                let gate: f64 = row.iter().zip(&lo).map(|(w, l)| f64::from(*w) * l).sum();
                sum += xn[c * n + i] * sigmoid(gate);
            }
            m.push(sum / hc as f64);
        }
    }
    (m, inj)
}

/// The fused read of the stream mixer (`read_down`, `read_up`) with its three matrices in
/// every weight type: the mixed input and the inject values are those of the f64 formulas
/// over the decoded weights, and a quantized type gives the bits of its decoded copy in F32,
/// except for the last bit of a weight of a type with an offset (`d * q + m`, which the device
/// may fuse). The shapes are the model's (four streams of 2560, rank 320) where a row of `up`
/// is whole blocks of the type, else rank 256 over streams of 512; one row and five.
#[test]
fn the_fused_read_takes_every_weight_type() {
    let Some(gpus) = gpus() else { return };
    let offset_types =
        [GgmlType::Q4_1, GgmlType::Q5_1, GgmlType::Q2_K, GgmlType::Q4_K, GgmlType::Q5_K];
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x3a91_c4f7_0b2e_6d15);
        eprintln!("  {}", gpu.name());
        for ty in TYPES {
            let bv = ty.block_values() as usize;
            let (hc, n, rank) = if 320 % bv == 0 { (4, 2560, 320) } else { (4, 512, 256) };
            let d = hc * n;
            let (mut worst, mut differ) = (0f64, 0usize);
            for rows in [1usize, 5] {
                let down = random_blocks(&mut rng, ty, rank * d / bv);
                let inject = random_blocks(&mut rng, ty, hc * d / bv);
                let up = random_blocks(&mut rng, ty, d * rank / bv);
                let decode = |bytes: &[u8], values: usize| -> Vec<f32> {
                    let mut out = vec![0f32; values];
                    ares_quant::dequantize(ty, bytes, &mut out).unwrap();
                    out
                };
                let (down_w, inject_w, up_w) =
                    (decode(&down, rank * d), decode(&inject, hc * d), decode(&up, d * rank));
                let r = rng.floats(rows * d, 3.0);
                let norm = rng.floats(d, 1.5);
                let (r_b, norm_b) = (gpu.floats(&r), gpu.floats(&norm));
                let act_bytes = quantized_activation_bytes((rows * n) as u64);
                let run = |ty: GgmlType, down: &[u8], inject: &[u8], up: &[u8]| {
                    let (down_b, inject_b, up_b) =
                        (gpu.bytes(down), gpu.bytes(inject), gpu.bytes(up));
                    let xn = gpu.empty((rows * d * 4) as u64);
                    let partials = gpu
                        .empty(Q4Kernels::read_partials(rows as u32, hc as u32, rank as u32) * 4);
                    let act = gpu.empty(act_bytes);
                    let inj = gpu.empty((rows * hc * 4) as u64);
                    let matrix = |b: &DeviceBuffer| MixerMatrix { ptr: b.ptr(), ty };
                    unsafe {
                        gpu.q4
                            .read_down(
                                &gpu.stream,
                                r_b.ptr(),
                                r_b.ptr(),
                                None,
                                Some((
                                    norm_b.ptr(),
                                    xn.ptr(),
                                    matrix(&down_b),
                                    Some(matrix(&inject_b)),
                                )),
                                partials.ptr(),
                                rows as u32,
                                hc as u32,
                                n as u32,
                                rank as u32,
                                EPS,
                            )
                            .unwrap();
                        gpu.q4
                            .read_up(
                                &gpu.stream,
                                xn.ptr(),
                                partials.ptr(),
                                matrix(&up_b),
                                act.ptr(),
                                Some(inj.ptr()),
                                None,
                                rows as u32,
                                hc as u32,
                                n as u32,
                                rank as u32,
                            )
                            .unwrap();
                    }
                    let m = gpu.read_at(&act, act_bytes - (rows * n * 4) as u64, rows * n);
                    (m, gpu.read(&inj, rows * hc))
                };
                let (m, inj) = run(ty, &down, &inject, &up);
                let (want_m, want_inj) =
                    read_reference(&r, &norm, &down_w, &inject_w, &up_w, (rows, hc, n, rank));
                worst = worst.max(rel_max(&m, &want_m)).max(rel_max(&inj, &want_inj));
                if ty != GgmlType::F32 {
                    let f32_bytes =
                        |w: &[f32]| -> Vec<u8> { w.iter().flat_map(|v| v.to_le_bytes()).collect() };
                    let (cm, cinj) = run(
                        GgmlType::F32,
                        &f32_bytes(&down_w),
                        &f32_bytes(&inject_w),
                        &f32_bytes(&up_w),
                    );
                    differ += m
                        .iter()
                        .chain(&inj)
                        .zip(cm.iter().chain(&cinj))
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
            }
            report(&format!("fused read {ty} rel_max ({differ} values differ from F32)"), worst);
            // The sums run over 10240 values in f32: 6e-5 of the largest output for IQ4_XS,
            // whose weights are large, with the same bits as the F32 copy gives.
            assert!(worst < 2e-4, "{ty}: {worst}");
            if !offset_types.contains(&ty) {
                assert_eq!(
                    differ, 0,
                    "{ty}: the read differs from that of the decoded copy in F32"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The indexer.

const ROPE: RopeParams = RopeParams { dim: 64, freq_base: 10_000_000.0, sections: [11, 11, 10, 0] };

fn indexer_shape(top_k: u32) -> IndexerShape {
    IndexerShape { n_head: 4, key_dim: 128, block: 4, top_k, rope_dim: ROPE.dim }
}

/// `(cos, sin)` of every rotated pair of a text cell.
fn rope_pairs(cell: usize) -> Vec<(f64, f64)> {
    (0..ROPE.dim as usize / 2)
        .map(|i| {
            let freq = f64::from(ROPE.freq_base).powf(-2.0 * i as f64 / f64::from(ROPE.dim));
            let angle = cell as f64 * freq;
            (angle.cos(), angle.sin())
        })
        .collect()
}

/// `rope(rms(x) * w)` at `cell`.
fn norm_rope(x: &[f32], w: &[f32], cell: usize) -> Vec<f64> {
    let normed = to_f32(&rms_norm(x, w, EPS));
    let mut out: Vec<f64> = normed.iter().map(|&v| f64::from(v)).collect();
    let pairs = rope_pairs(cell);
    let half = pairs.len();
    for (i, &(cos, sin)) in pairs.iter().enumerate() {
        let (x0, x1) = (f64::from(normed[i]), f64::from(normed[i + half]));
        out[i] = x0 * cos - x1 * sin;
        out[i + half] = x0 * sin + x1 * cos;
    }
    out
}

/// The pooled key of a block from the raw keys of its cells.
fn pool_key(raw: &[&[f32]], w: &[f32], first_cell: usize) -> Vec<f64> {
    let id = w.len();
    let mean: Vec<f32> = (0..id)
        .map(|i| (raw.iter().map(|r| f64::from(r[i])).sum::<f64>() / raw.len() as f64) as f32)
        .collect();
    norm_rope(&mean, w, first_cell)
}

fn half_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| {
            let h = u16::from_le_bytes(*b);
            let (sign, exp, frac) = (h >> 15, (h >> 10) & 31, f32::from(h & 1023));
            let value = match exp {
                0 => frac * 2.0f32.powi(-24),
                31 => f32::INFINITY,
                e => (1.0 + frac / 1024.0) * 2.0f32.powi(i32::from(e) - 15),
            };
            if sign != 0 { -value } else { value }
        })
        .collect()
}

/// A sequence as the test's oracle keeps it: the raw key of every cell.
struct HostSeq {
    slot: u32,
    table: u32,
    raw: Vec<Vec<f32>>,
}

/// Runs steps of several sequences through the window and the commit and compares the pooled
/// keys in the pool and the carried state with the oracle after every step.
fn indexer_protocol(gpu: &Gpu, format: KeyFormat) -> (f64, f64, f64) {
    let shape = indexer_shape(2048);
    let (id, ih, block) = (shape.key_dim as usize, shape.n_head as usize, shape.block as usize);
    let half = ROPE.dim as usize / 2;
    let mut rng = Rng(0x0123_4567_89ab_cdef ^ format as u64);
    let q_norm = rng.floats(id, 1.5);
    let k_norm = rng.floats(id, 1.5);
    let (q_norm_b, k_norm_b) = (gpu.floats(&q_norm), gpu.floats(&k_norm));

    // Three sequences with tables of 8 entries each; the key slots are scattered in the pool.
    let entries = 8u32;
    let n_seq = 3usize;
    let mut slots: Vec<u32> = (0..entries * n_seq as u32).collect();
    for i in (1..slots.len()).rev() {
        slots.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let key_table = gpu.words(&slots);
    let slot_bytes = shape.key_slot_bytes(format);
    let pool = gpu.empty(slot_bytes * u64::from(entries) * n_seq as u64);
    let carry_stride = shape.carry_len() + 6; // some slack between the slots
    let carry = gpu.empty(carry_stride * 4 * n_seq as u64);
    let mut seqs: Vec<HostSeq> = (0..n_seq)
        .map(|i| HostSeq {
            slot: (n_seq - 1 - i) as u32,
            table: i as u32 * entries,
            raw: Vec::new(),
        })
        .collect();
    let (mut worst_u, mut worst_pool, mut worst_carry) = (0.0f64, 0.0f64, 0.0f64);

    // Per step: for each sequence the rows of its window, whether it is direct, and the rows
    // accepted. Lengths are chosen so that blocks begin before a step and end in a later one,
    // that a rejection cuts a block open again, and that a window completes several blocks.
    let steps: [[(usize, bool, usize); 3]; 6] = [
        [(1, true, 1), (6, false, 2), (37, true, 37)],
        [(1, true, 1), (3, false, 3), (5, false, 0)],
        [(4, false, 1), (1, true, 1), (2, true, 2)],
        [(9, false, 9), (8, false, 5), (1, false, 1)],
        [(2, true, 2), (0, false, 0), (130, true, 130)],
        [(1, false, 1), (7, false, 7), (3, false, 2)],
    ];
    for step_plan in steps {
        let mut segments = Vec::new();
        let mut attn_rows = Vec::new();
        let (mut iq, mut raw) = (Vec::new(), Vec::new());
        for (seq, &(rows, direct, _)) in seqs.iter().zip(&step_plan) {
            if rows == 0 {
                continue;
            }
            let c0 = seq.raw.len();
            segments.push(Segment {
                row0: attn_rows.len() as u32,
                n_rows: rows as u32,
                slot: seq.slot,
                pos0: c0 as i32,
                direct,
            });
            for t in 0..rows {
                attn_rows.push(AttnRow {
                    table: seq.table,
                    pos: (c0 + t) as u32,
                    slot_base: 0,
                    n_slots: 0,
                });
                iq.extend(rng.floats(ih * id, 2.0));
                raw.extend(rng.floats(id, 2.0));
            }
        }
        let n_rows = attn_rows.len();
        let mut record = StepRecord::new(&segments, &vec![0u32; n_rows], None, &[]).unwrap();
        let dims = record.dims();
        let step_b = gpu.bytes(record.as_bytes());
        let rows_b = gpu.bytes(&AttnRow::to_bytes(&attn_rows));
        let rope_b = gpu.empty((n_rows * half * 8) as u64);
        let (iq_b, raw_b) = (gpu.floats(&iq), gpu.floats(&raw));
        let u_b = gpu.empty((n_rows * ih * id * 4) as u64);
        unsafe {
            gpu.layers.rope_table(&gpu.stream, step_b.ptr(), dims, rope_b.ptr(), ROPE).unwrap();
            gpu.q4
                .indexer_window(
                    &gpu.stream,
                    step_b.ptr(),
                    dims,
                    shape,
                    format,
                    EPS,
                    rows_b.ptr(),
                    iq_b.ptr(),
                    raw_b.ptr(),
                    q_norm_b.ptr(),
                    k_norm_b.ptr(),
                    rope_b.ptr(),
                    u_b.ptr(),
                    carry.ptr(),
                    carry_stride,
                    key_table.ptr(),
                    pool.ptr(),
                )
                .unwrap();
            // Direct segments commit right after their window.
            gpu.q4
                .indexer_commit(
                    &gpu.stream,
                    step_b.ptr(),
                    dims,
                    shape,
                    CommitSet::Direct,
                    rows_b.ptr(),
                    raw_b.ptr(),
                    rope_b.ptr(),
                    carry.ptr(),
                    carry_stride,
                )
                .unwrap();
        }
        // The queries.
        let u = gpu.read(&u_b, n_rows * ih * id);
        for (row, attn_row) in attn_rows.iter().enumerate() {
            for e in 0..ih {
                let at = (row * ih + e) * id;
                let want = norm_rope(&iq[at..at + id], &q_norm, attn_row.pos as usize);
                worst_u = worst_u.max(rel_max(&u[at..at + id], &want));
            }
        }
        // The verifier's verdict, then the deferred commit.
        let mut seg = 0;
        for (seq, &(rows, direct, accepted)) in seqs.iter_mut().zip(&step_plan) {
            if rows == 0 {
                continue;
            }
            if !direct {
                record.set_accepted(seg, accepted as u32).unwrap();
            }
            let row0 = segments[seg].row0 as usize;
            for t in 0..accepted {
                seq.raw.push(raw[(row0 + t) * id..(row0 + t + 1) * id].to_vec());
            }
            seg += 1;
        }
        let step_b = gpu.bytes(record.as_bytes());
        unsafe {
            gpu.q4
                .indexer_commit(
                    &gpu.stream,
                    step_b.ptr(),
                    dims,
                    shape,
                    CommitSet::Deferred,
                    rows_b.ptr(),
                    raw_b.ptr(),
                    rope_b.ptr(),
                    carry.ptr(),
                    carry_stride,
                )
                .unwrap();
        }
        // The pool holds the key of every complete block; the carried state the unfinished one.
        let pool_bytes = gpu.read_bytes(&pool, 0, pool.len() as usize);
        let carried = gpu.read(&carry, carry_stride as usize * n_seq);
        for seq in &seqs {
            let cells = seq.raw.len();
            for j in 0..cells / block {
                let members: Vec<&[f32]> =
                    (0..block).map(|m| seq.raw[j * block + m].as_slice()).collect();
                let want = pool_key(&members, &k_norm, j * block);
                let slot = slots[seq.table as usize + j / 32] as usize;
                let at = slot * slot_bytes as usize + (j % 32) * id * format.bytes() as usize;
                let got = match format {
                    KeyFormat::F32 => pool_bytes[at..at + id * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                    KeyFormat::F16 => half_to_f32(&pool_bytes[at..at + id * 2]),
                };
                worst_pool = worst_pool.max(rel_max(&got, &want));
            }
            let first = cells / block * block;
            let base = seq.slot as usize * carry_stride as usize;
            for cell in first..cells {
                let got = &carried[base + (cell - first) * id..][..id];
                assert_eq!(got, seq.raw[cell].as_slice(), "carried raw key of cell {cell}");
            }
            if cells > first {
                let pairs: Vec<f64> = rope_pairs(first).iter().flat_map(|&(c, s)| [c, s]).collect();
                let got = &carried[base + (block - 1) * id..][..2 * half];
                worst_carry = worst_carry.max(rel_max(got, &pairs));
            }
        }
    }
    (worst_u, worst_pool, worst_carry)
}

#[test]
fn the_indexer_window_and_commit_follow_the_sequences() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        eprintln!("  {}", gpu.name());
        for format in [KeyFormat::F32, KeyFormat::F16] {
            let (u, pool, carry) = indexer_protocol(&gpu, format);
            report(&format!("indexer queries rel_max ({format:?} pool)"), u);
            report(&format!("pooled keys rel_max ({format:?} pool)"), pool);
            report("carried rotation rel_max", carry);
            assert!(u < 2e-6 && carry < 1e-6, "{u} {carry}");
            // f16 rounds to 11 bits: half a unit in the last place of the largest value's binade.
            assert!(pool < if format == KeyFormat::F32 { 2e-6 } else { 6e-4 }, "{pool}");
        }
    }
}

// ---------------------------------------------------------------------------
// Block scores and the selection.

/// The cells a row attends to, by the rule of the architecture document: the transcription of
/// `ares_model`'s `select_cells`.
fn select_cells(rule: SelectionRule, shape: &IndexerShape, n: usize, scores: &[f32]) -> Vec<u32> {
    let (block, limit) = (shape.block as usize, shape.dense_limit() as usize);
    if n <= limit {
        return (0..n as u32).collect();
    }
    let (blocks, tail) = (n / block, n % block);
    let mut order: Vec<u32> = (0..blocks as u32).collect();
    order.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
    let budget = match rule {
        SelectionRule::CellBudget => limit - tail,
        SelectionRule::WholeBlocks => shape.top_k as usize,
    };
    let mut cells = Vec::new();
    for &j in &order {
        let take = block.min(budget - cells.len());
        if take == 0 {
            break;
        }
        cells.extend((0..take as u32).map(|i| j * block as u32 + i));
    }
    cells.extend((blocks * block) as u32..n as u32);
    cells.sort_unstable();
    cells
}

/// The cells of a list of `(start, end, before)` triples, with the checks every list must pass.
fn cells_of(list: &[u32], cells_claimed: u32) -> Vec<u32> {
    let mut cells = Vec::new();
    for triple in list.as_chunks::<3>().0 {
        assert!(triple[0] < triple[1], "an empty range {triple:?}");
        assert_eq!(triple[2] as usize, cells.len(), "the cells before a range");
        if let Some(&last) = cells.last() {
            assert!(
                triple[0] > last,
                "ranges ascend and do not touch needlessly: {triple:?} after {last}"
            );
        }
        cells.extend(triple[0]..triple[1]);
    }
    assert_eq!(cells.len(), cells_claimed as usize);
    cells
}

struct Selected {
    /// Per row the converted list and the exact list, as cells.
    cells: Vec<[Vec<u32>; 2]>,
    /// Per row the two lists as triples (converted, then exact).
    triples: Vec<[Vec<u32>; 2]>,
}

#[allow(clippy::too_many_arguments)]
fn run_select(
    gpu: &Gpu,
    shape: IndexerShape,
    rule: SelectionRule,
    attn_rows: &[AttnRow],
    scores: &DeviceBuffer,
    stride: u32,
    tables: Option<&DeviceBuffer>,
    cap: u32,
) -> Selected {
    let n = attn_rows.len();
    let rows_b = gpu.bytes(&AttnRow::to_bytes(attn_rows));
    let lists = gpu.empty(n as u64 * u64::from(cap) * 12);
    let counts = gpu.empty(n as u64 * 16);
    unsafe {
        gpu.q4
            .select(
                &gpu.stream,
                shape,
                rule,
                rows_b.ptr(),
                scores.ptr(),
                tables.map(DeviceBuffer::ptr),
                lists.ptr(),
                counts.ptr(),
                0,
                n as u32,
                stride,
                cap,
            )
            .unwrap();
    }
    let lists = gpu.read_words(&lists, n * cap as usize * 3);
    let counts = gpu.read_words(&counts, n * 4);
    let mut out = Selected { cells: Vec::new(), triples: Vec::new() };
    for row in 0..n {
        let c = &counts[row * 4..row * 4 + 4];
        assert!(c[0] + c[2] <= cap, "row {row} needs {} ranges of {cap}", c[0] + c[2]);
        let list = &lists[row * cap as usize * 3..];
        let conv = list[..c[0] as usize * 3].to_vec();
        let exact = list[c[0] as usize * 3..(c[0] + c[2]) as usize * 3].to_vec();
        out.cells.push([cells_of(&conv, c[1]), cells_of(&exact, c[3])]);
        out.triples.push([conv, exact]);
    }
    out
}

#[test]
fn block_scores_match_and_the_selection_follows_the_rule() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        eprintln!("  {}", gpu.name());
        let mut rng = Rng(0x7e57_5e1e_c7ed_0001);
        for (top_k, cells, format) in [
            (8u32, 300usize, KeyFormat::F32),
            (2048, 9000, KeyFormat::F32),
            (2048, 9000, KeyFormat::F16),
            (2048, 40_000, KeyFormat::F32),
            // The most blocks the selection takes: 65 536.
            (2048, 262_144, KeyFormat::F16),
        ] {
            let shape = indexer_shape(top_k);
            let (id, ih, block) =
                (shape.key_dim as usize, shape.n_head as usize, shape.block as usize);
            let n_blocks = cells / block;
            let entries = cells.div_ceil(128);
            // Pooled keys with a spread of norms, so that scores are spread too; the slots of
            // the entries are in reverse order in the pool.
            let keys: Vec<f32> = (0..n_blocks)
                .flat_map(|_| {
                    let scale = 0.2 + rng.unit().abs() * 2.0;
                    (0..id).map(|_| rng.unit() * scale).collect::<Vec<f32>>()
                })
                .collect();
            let stored: Vec<f32> = match format {
                KeyFormat::F32 => keys.clone(),
                KeyFormat::F16 => half_to_f32(&f32_to_half_bytes(&keys)),
            };
            let slots: Vec<u32> = (0..entries as u32).rev().collect();
            let slot_bytes = shape.key_slot_bytes(format) as usize;
            let mut pool = vec![0u8; entries * slot_bytes];
            for j in 0..n_blocks {
                let key = &keys[j * id..(j + 1) * id];
                let bytes = match format {
                    KeyFormat::F32 => key.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                    KeyFormat::F16 => f32_to_half_bytes(key),
                };
                let at = slots[j / 32] as usize * slot_bytes + (j % 32) * bytes.len();
                pool[at..at + bytes.len()].copy_from_slice(&bytes);
            }
            let (pool_b, table_b) = (gpu.bytes(&pool), gpu.words(&slots));

            // Rows at the dense limit, just past it with every tail, and at the end.
            let limit = shape.dense_limit() as usize;
            let mut positions: Vec<usize> =
                vec![0, limit - 1, limit, limit + 1, limit + 2, limit + 3];
            positions.extend([cells - 4, cells - 3, cells - 2, cells - 1]);
            positions.extend((0..6).map(|_| limit + rng.below((cells - limit) as u64) as usize));
            let attn_rows: Vec<AttnRow> = positions
                .iter()
                .map(|&p| AttnRow { table: 0, pos: p as u32, slot_base: 0, n_slots: 0 })
                .collect();
            let n = attn_rows.len();
            let u = rng.floats(n * ih * id, 1.0);
            let (u_b, rows_b) = (gpu.floats(&u), gpu.bytes(&AttnRow::to_bytes(&attn_rows)));
            let stride = n_blocks as u32 + 3;
            let scores_b = gpu.empty(n as u64 * u64::from(stride) * 4);
            unsafe {
                gpu.q4
                    .block_scores(
                        &gpu.stream,
                        shape,
                        format,
                        rows_b.ptr(),
                        u_b.ptr(),
                        table_b.ptr(),
                        pool_b.ptr(),
                        scores_b.ptr(),
                        0,
                        n as u32,
                        stride,
                    )
                    .unwrap();
            }
            let scores = gpu.read(&scores_b, n * stride as usize);
            let mut worst = 0.0f64;
            let mut want_scores = vec![Vec::new(); n];
            let mut moved_by_f16 = (0usize, 0usize);
            for (row, &p) in positions.iter().enumerate() {
                let visible = (p + 1) / block;
                if p < limit {
                    continue;
                }
                let scores_with = |keys: &[f32]| -> Vec<f64> {
                    (0..visible)
                        .map(|j| {
                            (0..ih)
                                .map(|e| {
                                    let dot: f64 = (0..id)
                                        .map(|i| {
                                            f64::from(u[(row * ih + e) * id + i])
                                                * f64::from(keys[j * id + i])
                                        })
                                        .sum();
                                    dot.max(0.0)
                                })
                                .sum()
                        })
                        .collect()
                };
                let want = scores_with(&stored);
                let got = &scores[row * stride as usize..][..visible];
                worst = worst.max(rel_max(got, &want));
                if format == KeyFormat::F16 {
                    // What the rounding of the keys to f16 does to the selection.
                    let rule = SelectionRule::CellBudget;
                    let as_f32 = |x: &[f64]| x.iter().map(|&s| s as f32).collect::<Vec<f32>>();
                    let rounded = select_cells(rule, &shape, p + 1, &as_f32(&want));
                    let exact = select_cells(rule, &shape, p + 1, &as_f32(&scores_with(&keys)));
                    moved_by_f16.0 +=
                        rounded.iter().filter(|c| exact.binary_search(c).is_err()).count();
                    moved_by_f16.1 += rounded.len();
                }
                want_scores[row] = want;
            }
            if format == KeyFormat::F16 {
                eprintln!(
                    "    f16 keys instead of f32: {} of {} selected cells differ ({cells} cells)",
                    moved_by_f16.0, moved_by_f16.1
                );
            }
            report(&format!("block scores rel_max, {cells} cells, {format:?} keys"), worst);
            assert!(worst < 3e-6, "{worst}");

            // The selection of the device's own scores is the reference's selection of them,
            // under both rules; from the exact scores it differs only where scores are within
            // the error above.
            for rule in [SelectionRule::CellBudget, SelectionRule::WholeBlocks] {
                let cap = shape.max_ranges(0);
                let selected =
                    run_select(&gpu, shape, rule, &attn_rows, &scores_b, stride, None, cap);
                let mut differ = 0usize;
                for (row, &p) in positions.iter().enumerate() {
                    let row_scores = &scores[row * stride as usize..][..(p + 1) / block];
                    let want = select_cells(rule, &shape, p + 1, row_scores);
                    assert_eq!(selected.cells[row][0], want, "row at cell {p}, {rule:?}");
                    assert!(selected.cells[row][1].is_empty());
                    if p + 1 > limit {
                        let exact: Vec<f32> = want_scores[row].iter().map(|&s| s as f32).collect();
                        let ideal = select_cells(rule, &shape, p + 1, &exact);
                        differ += want.iter().filter(|c| ideal.binary_search(c).is_err()).count();
                    }
                }
                report(
                    &format!("cells that differ from the selection by f64 scores, {rule:?}"),
                    differ as f64,
                );
            }
        }
    }
}

fn f32_to_half_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&v| {
            // Round to nearest even through the bits; the test values are far from the limits.
            let bits = v.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let frac = bits & 0x7f_ffff;
            let half = if exp <= 0 {
                // Subnormal half: shift the significand (with its leading one) down.
                let shift = 14 - exp;
                if shift > 24 {
                    0
                } else {
                    let sig = frac | 0x80_0000;
                    let (q, rest, halfway) =
                        (sig >> shift, sig & ((1 << shift) - 1), 1u32 << (shift - 1));
                    (q + u32::from(rest > halfway || (rest == halfway && q & 1 == 1))) as u16
                }
            } else {
                let (q, rest) = (((exp as u32) << 10) | (frac >> 13), frac & 0x1fff);
                (q + u32::from(rest > 0x1000 || (rest == 0x1000 && q & 1 == 1))) as u16
            };
            (sign | half).to_le_bytes()
        })
        .collect()
}

#[test]
fn the_selection_breaks_ties_by_index_and_splits_lists_by_entry_kind() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x0dd5_ba11_5eed_7777);
        for top_k in [8u32, 64, 2048] {
            let shape = indexer_shape(top_k);
            let block = shape.block as usize;
            let cells = shape.dense_limit() as usize * 3 + 700;
            let n_blocks = cells / block;
            let entries = cells.div_ceil(128);
            // Tables of mixed kinds: converted entries (bit 31) with exact ones at both ends and
            // one in the middle.
            let kinds: Vec<bool> =
                (0..entries).map(|e| !(e == 0 || e == entries / 2 || e + 2 >= entries)).collect();
            let table: Vec<u32> = kinds
                .iter()
                .enumerate()
                .map(|(e, &c)| e as u32 | if c { 1 << 31 } else { 0 })
                .collect();
            let kind_changes = kinds.windows(2).filter(|w| w[0] != w[1]).count() as u32;
            let table_b = gpu.words(&table);
            // Scores with many ties: a few levels only, zeros included, and a row of all zeros.
            let positions: Vec<usize> = (0..12).map(|i| cells - 1 - i).collect();
            let attn_rows: Vec<AttnRow> = positions
                .iter()
                .map(|&p| AttnRow { table: 0, pos: p as u32, slot_base: 0, n_slots: 0 })
                .collect();
            let n = attn_rows.len();
            let stride = n_blocks as u32;
            let mut scores = vec![0.0f32; n * n_blocks];
            for (row, chunk) in scores.chunks_exact_mut(n_blocks).enumerate() {
                let levels = [1u64, 2, 3, 5, 40, 1000][row % 6];
                for s in chunk.iter_mut() {
                    *s = if row == 7 { 0.0 } else { rng.below(levels) as f32 * 0.25 };
                }
            }
            let scores_b = gpu.floats(&scores);
            for rule in [SelectionRule::CellBudget, SelectionRule::WholeBlocks] {
                let cap = shape.max_ranges(kind_changes);
                let selected = run_select(
                    &gpu,
                    shape,
                    rule,
                    &attn_rows,
                    &scores_b,
                    stride,
                    Some(&table_b),
                    cap,
                );
                for (row, &p) in positions.iter().enumerate() {
                    let want = select_cells(
                        rule,
                        &shape,
                        p + 1,
                        &scores[row * n_blocks..][..(p + 1) / block],
                    );
                    let [conv, exact] = &selected.cells[row];
                    assert!(
                        conv.iter().all(|&c| kinds[c as usize / 128]),
                        "a cell of an exact entry"
                    );
                    assert!(
                        exact.iter().all(|&c| !kinds[c as usize / 128]),
                        "a cell of a converted entry"
                    );
                    let mut all: Vec<u32> = conv.iter().chain(exact).copied().collect();
                    all.sort_unstable();
                    assert_eq!(all, want, "top_k {top_k}, row at cell {p}, {rule:?}");
                    // No range crosses from an entry of one kind into one of the other.
                    for triple in selected.triples[row].iter().flat_map(|l| l.as_chunks::<3>().0) {
                        let (a, b) = (triple[0] as usize / 128, (triple[1] as usize - 1) / 128);
                        assert!(kinds[a..=b].iter().all(|&k| k == kinds[a]));
                    }
                }
                // A capacity that is too small is reported, not overrun.
                let small = 2;
                let rows_b = gpu.bytes(&AttnRow::to_bytes(&attn_rows));
                let lists = gpu.empty(n as u64 * small * 12 + 64);
                let counts = gpu.empty(n as u64 * 16);
                unsafe {
                    gpu.q4
                        .select(
                            &gpu.stream,
                            shape,
                            rule,
                            rows_b.ptr(),
                            scores_b.ptr(),
                            Some(table_b.ptr()),
                            lists.ptr(),
                            counts.ptr(),
                            0,
                            n as u32,
                            stride,
                            small as u32,
                        )
                        .unwrap();
                }
                let counts = gpu.read_words(&counts, n * 4);
                for (row, c) in counts.as_chunks::<4>().0.iter().enumerate() {
                    let [conv, exact] = &selected.triples[row];
                    assert_eq!([c[0] as usize, c[2] as usize], [conv.len() / 3, exact.len() / 3]);
                }
                assert!(counts.as_chunks::<4>().0.iter().any(|c| u64::from(c[0] + c[2]) > small));
                let guard = gpu.read_bytes(&lists, n as u64 * small * 12, 64);
                assert!(guard.iter().all(|&b| b == 0xFF), "the lists were overrun");
            }
        }
        eprintln!("  {}: ties, kinds and capacity hold", gpu.name());
    }
}

// ---------------------------------------------------------------------------
// The hand-over to the attention kernels.

/// `ITEM_CONVERTED` of the attention plan: the item reads converted entries.
const ITEM_CONVERTED: u32 = 1;

#[test]
fn selected_lists_reach_the_attention_kernels() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let attention = AttentionKernels::load(device).unwrap();
        let mut rng = Rng(0xa77e_17d0_0000_0b0b);
        // A small budget, so that every selected cell weighs enough to be seen in the output.
        let shape = indexer_shape(64);
        let (id, ih, block) = (shape.key_dim as usize, shape.n_head as usize, shape.block as usize);
        let (hd, kv_heads, g) = (128usize, 2usize, 4usize);
        let geo = CacheGeometry {
            k: CacheFormat::F16,
            v: CacheFormat::F16,
            head_dim: hd as u32,
            kv_heads: kv_heads as u32,
            kv_total: kv_heads as u32,
            first_kv_head: 0,
            q_per_kv: g as u32,
        };
        let cells = 1500usize;
        let entries = cells.div_ceil(128);
        // The first entry and the two newest are exact, the others converted.
        let kinds: Vec<bool> = (0..entries).map(|e| !(e == 0 || e + 2 >= entries)).collect();
        let (mut n_conv, mut n_exact) = (0u32, 0u32);
        let table: Vec<u32> = kinds
            .iter()
            .map(|&converted| {
                let counter = if converted { &mut n_conv } else { &mut n_exact };
                *counter += 1;
                (*counter - 1) | if converted { 1 << 31 } else { 0 }
            })
            .collect();
        let table_b = gpu.words(&table);
        let exact_pool = gpu.empty(geo.exact_entry_bytes() * u64::from(n_exact));
        let conv_pool = gpu.empty(geo.converted_entry_bytes() * u64::from(n_conv));

        // The sequence's K and V, appended by the attention module's own kernel.
        let k = rng.floats(cells * kv_heads * hd, 1.0);
        let v = rng.floats(cells * kv_heads * hd, 1.0);
        let all_rows: Vec<AttnRow> = (0..cells as u32)
            .map(|pos| AttnRow { table: 0, pos, slot_base: 0, n_slots: 0 })
            .collect();
        let (k_b, v_b) = (gpu.floats(&k), gpu.floats(&v));
        let all_rows_b = gpu.bytes(&AttnRow::to_bytes(&all_rows));
        let status = gpu.words(&[0]);
        unsafe {
            attention
                .append(
                    &gpu.stream,
                    &geo,
                    all_rows_b.ptr(),
                    cells as u32,
                    k_b.ptr(),
                    v_b.ptr(),
                    table_b.ptr(),
                    exact_pool.ptr(),
                    conv_pool.ptr(),
                    ares_cuda::DevicePtr::NULL,
                    status.ptr(),
                )
                .unwrap();
        }
        assert_eq!(gpu.read_words(&status, 1), [0], "the append reported a problem");
        // What is stored is f16.
        let k16 = half_to_f32(&f32_to_half_bytes(&k));
        let v16 = half_to_f32(&f32_to_half_bytes(&v));

        // Pooled keys of every block, in a pool of their own.
        let n_blocks = cells / block;
        let keys = rng.floats(n_blocks * id, 1.0);
        let key_slots: Vec<u32> = (0..entries as u32).rev().collect();
        let slot_bytes = shape.key_slot_bytes(KeyFormat::F32) as usize;
        let mut pool = vec![0u8; entries * slot_bytes];
        for (j, key) in keys.chunks_exact(id).enumerate() {
            let at = key_slots[j / 32] as usize * slot_bytes + (j % 32) * id * 4;
            for (i, value) in key.iter().enumerate() {
                pool[at + i * 4..at + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        let (key_pool, key_table) = (gpu.bytes(&pool), gpu.words(&key_slots));

        // The rows of the step: the newest cells, some in the middle, one within the dense
        // limit (all of its cells are in the first, exact entry).
        let positions = [cells - 1, cells - 2, cells - 3, cells - 4, 700, 68, 40];
        let n = positions.len();
        let (pc, pe) = (3u32, 2u32);
        let pieces = pc + pe;
        let cap = shape.max_ranges(2);
        let heads = kv_heads as u32;
        let mut plan = Plan {
            rows: positions
                .iter()
                .enumerate()
                .map(|(row, &p)| AttnRow {
                    table: 0,
                    pos: p as u32,
                    slot_base: row as u32 * pieces,
                    n_slots: pieces,
                })
                .collect(),
            row_ids: (0..n as u32).collect(),
            ranges: vec![0; n * cap as usize * 3],
            state_slots: n as u32 * pieces,
            needs_merge: true,
            ..Plan::default()
        };
        for (flags, count, slot0) in [(ITEM_CONVERTED, pc, 0), (0, pe, pc)] {
            let start = plan.items.len();
            for row in 0..n as u32 {
                for piece in 0..count {
                    for kv_head in 0..heads {
                        plan.items.push(AttnItem {
                            table: 0,
                            row0: row,
                            n_rows: 1,
                            kv_head,
                            flags,
                            slot: slot0 + piece,
                            ..AttnItem::default()
                        });
                    }
                }
            }
            plan.groups.push((AttnShape::for_vecs(g as u32), start..plan.items.len()));
        }
        let hands: Vec<ListHandOver> = (0..n as u32)
            .map(|row| ListHandOver {
                ranges_off: row * cap * 3,
                item_conv: row * pc * heads,
                pieces_conv: pc,
                item_exact: n as u32 * pc * heads + row * pe * heads,
                pieces_exact: pe,
                kv_heads: heads,
            })
            .collect();
        let layout = plan.layout();
        let mut plan_bytes = vec![0u8; layout.bytes as usize];
        plan.write(&mut plan_bytes);
        let plan_b = gpu.bytes(&plan_bytes);
        let buffers = unsafe { plan.on_device(plan_b.ptr()) };

        let u = rng.floats(n * ih * id, 1.0);
        let q = rng.floats(n * kv_heads * g * hd, 4.0);
        let (u_b, q_b) = (gpu.floats(&u), gpu.floats(&q));
        let stride = n_blocks as u32;
        let scores_b = gpu.empty(n as u64 * u64::from(stride) * 4);
        let counts = gpu.empty(n as u64 * 16);
        let hands_b = gpu.bytes(&ListHandOver::to_bytes(&hands));
        let out = gpu.empty((n * kv_heads * g * hd * 4) as u64);
        let scratch = gpu.empty(plan.state_bytes(&geo));
        let rows_ptr = plan_b.ptr().offset(layout.rows);
        let ranges_ptr = plan_b.ptr().offset(layout.ranges);
        unsafe {
            gpu.q4
                .block_scores(
                    &gpu.stream,
                    shape,
                    KeyFormat::F32,
                    rows_ptr,
                    u_b.ptr(),
                    key_table.ptr(),
                    key_pool.ptr(),
                    scores_b.ptr(),
                    0,
                    n as u32,
                    stride,
                )
                .unwrap();
            gpu.q4
                .select(
                    &gpu.stream,
                    shape,
                    SelectionRule::CellBudget,
                    rows_ptr,
                    scores_b.ptr(),
                    Some(table_b.ptr()),
                    ranges_ptr,
                    counts.ptr(),
                    0,
                    n as u32,
                    stride,
                    cap,
                )
                .unwrap();
            gpu.q4
                .hand_over(
                    &gpu.stream,
                    hands_b.ptr(),
                    counts.ptr(),
                    ranges_ptr,
                    plan_b.ptr().offset(layout.items),
                    n as u32,
                )
                .unwrap();
            attention
                .attend(
                    &gpu.stream,
                    &geo,
                    &buffers,
                    &AttendBuffers {
                        tables: table_b.ptr(),
                        exact_pool: exact_pool.ptr(),
                        converted_pool: conv_pool.ptr(),
                        far_pool: ares_cuda::DevicePtr::NULL,
                        q: q_b.ptr(),
                        gate: None,
                        out: out.ptr(),
                        scratch: scratch.ptr(),
                    },
                    1.0 / (hd as f32).sqrt(),
                )
                .unwrap();
        }

        // The plan on the device: the pieces of a row cover exactly the cells the rule selects,
        // each in entries of its item's kind, the same for every KV head.
        let scores = gpu.read(&scores_b, n * stride as usize);
        let after = gpu.read_bytes(&plan_b, 0, layout.bytes as usize);
        let word = |at: usize| u32::from_le_bytes(after[at..at + 4].try_into().unwrap());
        let mut empty_pieces = 0;
        let mut selected = Vec::new();
        for (row, &p) in positions.iter().enumerate() {
            let want = select_cells(
                SelectionRule::CellBudget,
                &shape,
                p + 1,
                &scores[row * stride as usize..][..(p + 1) / block],
            );
            let mut got = Vec::new();
            for (kind, hand_item, count) in
                [(true, hands[row].item_conv, pc), (false, hands[row].item_exact, pe)]
            {
                for piece in 0..count {
                    let mut per_head = Vec::new();
                    for head in 0..heads {
                        let at = layout.items as usize
                            + (hand_item + piece * heads + head) as usize * 48;
                        let (off, n_ranges, total) = (word(at + 16), word(at + 20), word(at + 24));
                        let triples: Vec<u32> = (0..n_ranges * 3)
                            .map(|i| word(layout.ranges as usize + (off + i) as usize * 4))
                            .collect();
                        per_head.push(cells_of(&triples, total));
                    }
                    assert!(per_head.windows(2).all(|w| w[0] == w[1]));
                    assert!(per_head[0].iter().all(|&c| kinds[c as usize / 128] == kind));
                    empty_pieces += usize::from(per_head[0].is_empty());
                    got.extend_from_slice(&per_head[0]);
                }
            }
            got.sort_unstable();
            assert_eq!(got, want, "row at cell {p}");
            selected.push(want);
        }

        // The attention of the selected cells, in f64 over the stored f16 values.
        let got = gpu.read(&out, n * kv_heads * g * hd);
        let mut worst = 0.0f64;
        for (row, cells) in selected.iter().enumerate() {
            for head in 0..kv_heads * g {
                let b = head / g;
                let qh = &q[(row * kv_heads * g + head) * hd..][..hd];
                let logits: Vec<f64> = cells
                    .iter()
                    .map(|&c| {
                        let key = &k16[(c as usize * kv_heads + b) * hd..][..hd];
                        qh.iter().zip(key).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                let mut want = vec![0.0f64; hd];
                for (&c, w) in cells.iter().zip(&weights) {
                    let value = &v16[(c as usize * kv_heads + b) * hd..][..hd];
                    for (o, &x) in want.iter_mut().zip(value) {
                        *o += w / total * f64::from(x);
                    }
                }
                worst = worst.max(rel_max(&got[(row * kv_heads * g + head) * hd..][..hd], &want));
            }
        }
        eprintln!("  {}", gpu.name());
        report(
            &format!("attention over device-made lists rel_max ({empty_pieces} empty pieces)"),
            worst,
        );
        assert!(empty_pieces > 0, "the test should cover pieces without cells");
        assert!(worst < 6e-3, "{worst}");
    }
}

// ---------------------------------------------------------------------------
// The hashed n-gram embedding.

const NGRAM_REAL: NgramShape = NgramShape {
    n_stream: 4,
    n_embd: 2560,
    heads: 16,
    row_dim: 160,
    ty: GgmlType::IQ4_NL,
    conv_kernel: 4,
    dilation: 3,
};
const NGRAM_SMALL: NgramShape = NgramShape {
    n_stream: 3,
    n_embd: 96,
    heads: 3,
    row_dim: 32,
    ty: GgmlType::IQ4_NL,
    conv_kernel: 3,
    dilation: 2,
};

/// The table rows of the n-gram block in every weight type: rows of 256 values (every
/// type's block divides them), decoded as `ares_quant::dequantize` decodes them, up to the
/// last bit of a value of a type with an offset (`d * q + m`, which the device may fuse).
#[test]
fn the_ngram_rows_of_every_weight_type_are_decoded() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x51c3_0d2a_77e4_9b01);
        for ty in TYPES {
            let shape = NgramShape {
                n_stream: 2,
                n_embd: 512,
                heads: 2,
                row_dim: 256,
                ty,
                conv_kernel: 4,
                dilation: 3,
            };
            let rows = 3usize;
            let n = shape.n_embd as usize;
            let values = rows * n;
            let fetched = random_blocks(&mut rng, ty, values / ty.block_values() as usize);
            assert_eq!(fetched.len(), rows * shape.fetched_row_bytes() as usize);
            let mut want = vec![0.0f32; values];
            ares_quant::dequantize(ty, &fetched, &mut want).unwrap();
            let fetched_b = gpu.bytes(&fetched);
            let emb_b = gpu.empty((values * 4) as u64);
            unsafe {
                gpu.q4
                    .ngram_decode(&gpu.stream, shape, fetched_b.ptr(), emb_b.ptr(), rows as u32)
                    .unwrap();
            }
            let got = gpu.read(&emb_b, values);
            let scale = want.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 2e-6 * scale, "{ty} value {i}: {g} for {w}");
            }
        }
    }
}

#[test]
fn the_ngram_rows_are_decoded_and_gated() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x9e37_79b9_0000_4242);
        let (mut worst_decode, mut worst_gate, mut worst_gated, mut worst_nrm) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (shape, rows) in [(NGRAM_REAL, 1usize), (NGRAM_SMALL, 7), (NGRAM_REAL, 19)] {
            let (hc, n) = (shape.n_stream as usize, shape.n_embd as usize);
            let d = hc * n;
            // Fetched rows: IQ4_NL blocks with random scales and nibbles.
            let n_blocks = rows * n / 32;
            assert_eq!(n_blocks * 18, rows * shape.fetched_row_bytes() as usize);
            let mut fetched = Vec::with_capacity(n_blocks * 18);
            for _ in 0..n_blocks {
                fetched.extend(f32_to_half_bytes(&[rng.unit() * 0.02]));
                fetched.extend((0..16).map(|_| rng.next() as u8));
            }
            let mut want = vec![0.0f32; rows * n];
            ares_quant::dequantize(GgmlType::IQ4_NL, &fetched, &mut want).unwrap();
            let fetched_b = gpu.bytes(&fetched);
            let emb_b = gpu.empty((rows * n * 4) as u64);
            unsafe {
                gpu.q4
                    .ngram_decode(&gpu.stream, shape, fetched_b.ptr(), emb_b.ptr(), rows as u32)
                    .unwrap();
            }
            let want64: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
            worst_decode = worst_decode.max(rel_max(&gpu.read(&emb_b, rows * n), &want64));

            // The gate, with some streams whose key is nearly orthogonal to the residual.
            let key = rng.floats(rows * d, 2.0);
            let value = rng.floats(rows * n, 1.5);
            let r = rng.floats(rows * d, 3.0);
            let (nk, nq, nc) = (rng.floats(d, 1.5), rng.floats(d, 1.5), rng.floats(d, 1.5));
            let bufs = [&key, &value, &r, &nk, &nq, &nc].map(|x| gpu.floats(x));
            let gated_b = gpu.empty((rows * d * 4) as u64);
            let nrm_b = gpu.empty((rows * d * 4) as u64);
            let gate_b = gpu.empty((rows * hc * 4) as u64);
            unsafe {
                gpu.q4
                    .ngram_gate(
                        &gpu.stream,
                        shape,
                        EPS,
                        bufs[0].ptr(),
                        bufs[1].ptr(),
                        bufs[2].ptr(),
                        bufs[3].ptr(),
                        bufs[4].ptr(),
                        bufs[5].ptr(),
                        gated_b.ptr(),
                        nrm_b.ptr(),
                        Some(gate_b.ptr()),
                        rows as u32,
                    )
                    .unwrap();
            }
            let (mut want_gate, mut want_gated, mut want_nrm) =
                (Vec::new(), Vec::new(), Vec::new());
            for t in 0..rows {
                for c in 0..hc {
                    let (at, w) = (t * d + c * n, c * n);
                    let kn = to_f32(&rms_norm(&key[at..at + n], &nk[w..w + n], EPS));
                    let qn = to_f32(&rms_norm(&r[at..at + n], &nq[w..w + n], EPS));
                    let s =
                        kn.iter().zip(&qn).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>()
                            / (n as f64).sqrt();
                    let sign = if s > 0.0 {
                        1.0
                    } else if s < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let g = sigmoid(sign * s.abs().max(1e-6).sqrt()) as f32;
                    want_gate.push(f64::from(g));
                    let gated: Vec<f32> = value[t * n..(t + 1) * n]
                        .iter()
                        .map(|&v| (f64::from(v) * f64::from(g)) as f32)
                        .collect();
                    want_nrm.extend(rms_norm(&gated, &nc[w..w + n], EPS));
                    want_gated.extend(gated.iter().map(|&v| f64::from(v)));
                }
            }
            worst_gate = worst_gate.max(rel_max(&gpu.read(&gate_b, rows * hc), &want_gate));
            worst_gated = worst_gated.max(rel_max(&gpu.read(&gated_b, rows * d), &want_gated));
            worst_nrm = worst_nrm.max(rel_max(&gpu.read(&nrm_b, rows * d), &want_nrm));
        }
        eprintln!("  {}", gpu.name());
        report("n-gram decode rel_max", worst_decode);
        report("n-gram gate rel_max", worst_gate);
        report("n-gram gated value rel_max", worst_gated);
        report("n-gram normed gated value rel_max", worst_nrm);
        assert!(worst_decode == 0.0, "{worst_decode}");
        assert!(worst_gate < 2e-6 && worst_gated < 2e-6 && worst_nrm < 2e-6);
    }
}

/// Steps of two sequences through the window and the commit of the n-gram convolution; the
/// history on the device is compared with the oracle's after every step. Returns the largest
/// error of the residual.
fn ngram_protocol(gpu: &Gpu, shape: NgramShape, taps_type: GgmlType) -> f64 {
    let d = shape.wide() as usize;
    let (kernel, dilation, back) =
        (shape.conv_kernel as usize, shape.dilation as usize, shape.back() as usize);
    let mut rng = Rng(0x00c0_ffee_1234_5678 + u64::from(shape.n_embd));
    let taps = rng.floats(d * kernel, 1.0);
    let (taps_b, taps_used) = match taps_type {
        GgmlType::F16 => {
            let bytes = f32_to_half_bytes(&taps);
            (gpu.bytes(&bytes), half_to_f32(&bytes))
        }
        _ => (gpu.floats(&taps), taps.clone()),
    };
    let n_slots = 3usize;
    let stride = shape.history_len() + 8;
    let history = gpu.floats(&vec![0.0f32; stride as usize * n_slots]);
    // Per sequence its slot and the nrm of every committed cell.
    let mut seqs: Vec<(u32, Vec<Vec<f32>>)> = vec![(2, Vec::new()), (0, Vec::new())];
    let steps: [[(usize, bool, usize); 2]; 6] = [
        [(1, true, 1), (5, false, 2)],
        [(2, false, 2), (1, true, 1)],
        [(7, false, 3), (40, true, 40)],
        [(1, true, 1), (3, false, 0)],
        [(12, true, 12), (2, false, 1)],
        [(1, false, 1), (70, false, 70)],
    ];
    let mut worst = 0.0f64;
    for plan in steps {
        let mut segments = Vec::new();
        let mut n_rows = 0usize;
        for (seq, &(rows, direct, _)) in seqs.iter().zip(&plan) {
            segments.push(Segment {
                row0: n_rows as u32,
                n_rows: rows as u32,
                slot: seq.0,
                pos0: seq.1.len() as i32,
                direct,
            });
            n_rows += rows;
        }
        let mut record = StepRecord::new(&segments, &vec![0u32; n_rows], None, &[]).unwrap();
        let dims = record.dims();
        let r = rng.floats(n_rows * d, 3.0);
        let gated = rng.floats(n_rows * d, 1.0);
        let nrm = rng.floats(n_rows * d, 2.0);
        let (r_b, gated_b, nrm_b) = (gpu.floats(&r), gpu.floats(&gated), gpu.floats(&nrm));
        let step_b = gpu.bytes(record.as_bytes());
        unsafe {
            gpu.q4
                .ngram_window(
                    &gpu.stream,
                    step_b.ptr(),
                    dims,
                    shape,
                    r_b.ptr(),
                    gated_b.ptr(),
                    nrm_b.ptr(),
                    taps_b.ptr(),
                    taps_type,
                    history.ptr(),
                    stride,
                )
                .unwrap();
            gpu.q4
                .ngram_commit(
                    &gpu.stream,
                    step_b.ptr(),
                    dims,
                    shape,
                    CommitSet::Direct,
                    nrm_b.ptr(),
                    history.ptr(),
                    stride,
                )
                .unwrap();
        }
        // The residual, from the oracle's committed cells and the rows of the window.
        let got = gpu.read(&r_b, n_rows * d);
        let mut want = vec![0.0f64; n_rows * d];
        for (seg, (_, committed)) in segments.iter().zip(&seqs) {
            let row0 = seg.row0 as usize;
            for t in 0..seg.n_rows as usize {
                let cell = committed.len() + t;
                for ch in 0..d {
                    let mut conv = 0.0f64;
                    for k in 0..kernel {
                        let behind = (kernel - 1 - k) * dilation;
                        let x = if behind > cell {
                            0.0
                        } else if cell - behind < committed.len() {
                            committed[cell - behind][ch]
                        } else {
                            nrm[(row0 + cell - behind - committed.len()) * d + ch]
                        };
                        conv += f64::from(taps_used[ch * kernel + k]) * f64::from(x);
                    }
                    let at = (row0 + t) * d + ch;
                    want[at] = f64::from(r[at]) + f64::from(gated[at]) + silu(conv);
                }
            }
        }
        worst = worst.max(rel_max(&got, &want));
        // The verdict and the deferred commit.
        for (i, ((_, committed), &(_, direct, accepted))) in seqs.iter_mut().zip(&plan).enumerate()
        {
            if !direct {
                record.set_accepted(i, accepted as u32).unwrap();
            }
            let row0 = segments[i].row0 as usize;
            for t in 0..accepted {
                committed.push(nrm[(row0 + t) * d..(row0 + t + 1) * d].to_vec());
            }
        }
        let step_b = gpu.bytes(record.as_bytes());
        unsafe {
            gpu.q4
                .ngram_commit(
                    &gpu.stream,
                    step_b.ptr(),
                    dims,
                    shape,
                    CommitSet::Deferred,
                    nrm_b.ptr(),
                    history.ptr(),
                    stride,
                )
                .unwrap();
        }
        let device_history = gpu.read(&history, stride as usize * n_slots);
        for (slot, committed) in &seqs {
            for i in 0..back {
                let got = &device_history[*slot as usize * stride as usize + i * d..][..d];
                // Entry i is the cell `back - i` places before the next one.
                match committed.len().checked_sub(back - i) {
                    Some(cell) => {
                        assert_eq!(got, committed[cell].as_slice(), "history of cell {cell}")
                    }
                    None => assert!(got.iter().all(|&v| v == 0.0), "history before the start"),
                }
            }
        }
    }
    worst
}

#[test]
fn the_ngram_window_and_commit_follow_the_sequences() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        eprintln!("  {}", gpu.name());
        for (shape, taps_type) in
            [(NGRAM_SMALL, GgmlType::F32), (NGRAM_REAL, GgmlType::F16), (NGRAM_REAL, GgmlType::F32)]
        {
            let worst = ngram_protocol(&gpu, shape, taps_type);
            report(
                &format!("n-gram residual rel_max, width {}, {taps_type} taps", shape.wide()),
                worst,
            );
            assert!(worst < 2e-6, "{worst}");
        }
    }
}
