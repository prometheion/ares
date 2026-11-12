//! Tests of the layer kernels (`kernels/layers.cu`, `src/layers.rs`) that need
//! nothing but the quantization reference: the step record, the embedding
//! gather against `ares_quant::dequantize`, the norms, the fused quantization
//! (byte for byte against `Kernels::quantize_activations`), the rotary table and
//! the attention preparation against plain f64 formulas. The recurrent kernels
//! (convolution, delta rule, commit) are held to `ares_model`'s reference in
//! `crates/model/tests/layers_gpu.rs`, which needs that crate.
//!
//! Like the other GPU tests they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test layers
//! ```
//!
//! `ARES_TEST_GGUF` is not used here; the real-weight checks are in the model crate.

// The tests launch kernels on buffers they allocated a few lines earlier, with sizes taken from
// the same shapes; a comment on each of the launches would repeat that.
#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_cuda::{
    AttentionShape, Device, DeviceBuffer, DevicePtr, EmbedRows, KeptRows, Kernels, LayerKernels,
    NormKind, OutputGate, RopeParams, Segment, StepRecord, Stream, Wait, device_count,
    quantized_activation_bytes,
};

fn gpus() -> Option<&'static [Device]> {
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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

struct Gpu {
    device: Device,
    stream: Stream,
    layers: LayerKernels,
    kernels: Kernels,
}

impl Gpu {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            stream: Stream::new(device, 0).unwrap(),
            layers: LayerKernels::load(device).unwrap(),
            kernels: Kernels::load(device).unwrap(),
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
    fn empty(&self, bytes: u64) -> DeviceBuffer {
        let buffer = DeviceBuffer::new(&self.device, bytes.max(16)).unwrap();
        // Poison, so a kernel that does not write what it should is seen.
        buffer.write(0, &vec![0xFFu8; bytes as usize]).unwrap();
        buffer
    }
    fn read(&self, buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; n * 4];
        buffer.read(0, &mut bytes).unwrap();
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }
    fn read_bytes(&self, buffer: &DeviceBuffer, n: usize) -> Vec<u8> {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; n];
        buffer.read(0, &mut bytes).unwrap();
        bytes
    }
    fn step(&self, record: &StepRecord) -> DeviceBuffer {
        self.bytes(record.as_bytes())
    }
}

fn random_floats(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.unit() * scale).collect()
}

/// Largest absolute difference, relative to the largest value of `reference`.
fn rel_max(got: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(got.len(), reference.len());
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(reference).fold(0.0f64, |m, (g, r)| m.max((f64::from(*g) - r).abs())) / scale
}

// ---------------------------------------------------------------------------

#[test]
fn the_step_record_is_checked_and_laid_out() {
    let seg = |row0, n_rows, slot, direct| Segment { row0, n_rows, slot, pos0: 7, direct };
    let tokens = [10u32, 11, 12, 13, 14];
    let record =
        StepRecord::new(&[seg(0, 2, 5, true), seg(2, 3, 1, false)], &tokens, None, &[1, 4])
            .unwrap();
    let dims = record.dims();
    assert_eq!((dims.n_rows, dims.n_segs, dims.max_seg_rows, dims.n_out), (5, 2, 3, 2));
    let b = record.as_bytes();
    assert_eq!(b.len() % 16, 0);
    let word = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    assert_eq!((word(0), word(4), word(8)), (5, 2, 2));
    let (off_segs, off_tokens, off_pos, off_out) =
        (word(16) as usize, word(20) as usize, word(24) as usize, word(28) as usize);
    assert!([off_segs, off_tokens, off_pos, off_out].iter().all(|o| o % 16 == 0));
    // Segment 1: row0 2, 3 rows, slot 1, all accepted until told otherwise, position 7, not direct.
    let s1 = off_segs + 32;
    assert_eq!([word(s1), word(s1 + 4), word(s1 + 8), word(s1 + 12)], [2, 3, 1, 3]);
    assert_eq!((word(s1 + 16), word(s1 + 20)), (7, 0));
    assert_eq!(word(off_segs + 20), 1, "segment 0 is direct");
    assert_eq!(word(off_tokens + 12), 13);
    // Text positions: row 3 is the second row of segment 1: (8, 8, 8, 0).
    assert_eq!([0, 1, 2, 3].map(|k| word(off_pos + 3 * 16 + k * 4)), [8, 8, 8, 0]);
    assert_eq!((word(off_out), word(off_out + 4)), (1, 4));

    let mut record = record;
    record.set_accepted(1, 2).unwrap();
    assert_eq!(u32::from_le_bytes(record.as_bytes()[s1 + 12..s1 + 16].try_into().unwrap()), 2);
    assert!(record.set_accepted(1, 4).is_err());
    assert!(record.set_accepted(2, 0).is_err());
    assert_eq!(StepRecord::accepted_offset(1), (s1 + 12) as u64);

    // Overlap, overrun, a bad output row and a wrong number of positions are refused.
    assert!(
        StepRecord::new(&[seg(0, 3, 0, false), seg(2, 3, 1, false)], &tokens, None, &[]).is_err()
    );
    assert!(StepRecord::new(&[seg(3, 3, 0, false)], &tokens, None, &[]).is_err());
    assert!(StepRecord::new(&[seg(0, 5, 0, false)], &tokens, None, &[5]).is_err());
    assert!(StepRecord::new(&[seg(0, 5, 0, false)], &tokens, Some(&[[0; 4]; 4]), &[]).is_err());
    // An empty step is a valid record.
    let empty = StepRecord::new(&[], &[], None, &[]).unwrap();
    assert_eq!(empty.dims().n_rows, 0);
}

// ---------------------------------------------------------------------------

/// The blocks of the golden fixture of `ty` in `ares-quant` (`tests/golden/dequant_<type>.bin`),
/// which ggml's own code produced: blocks its quantizer made from float data, then raw random
/// ones whose index fields reach every entry of the type's codebooks and whose scales have
/// the values the format allows (f16 scales of the whole finite range, every E8M0 and UE4M3
/// byte, all the digits of a TQ1_0 byte). `ares_quant::dequantize` is held to ggml's values
/// for them bit for bit.
fn golden_blocks(ty: GgmlType) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../quant/tests/golden")
        .join(format!("dequant_{}.bin", ty.name()));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let word = |i: usize| u32::from_le_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap());
    assert_eq!(&bytes[..4], b"EQGD", "{}", path.display());
    assert_eq!((word(1), word(2), word(3)), (ty.id(), ty.block_values(), ty.block_bytes()));
    let n_blocks = word(4) as usize;
    bytes[32..32 + n_blocks * ty.block_bytes() as usize].to_vec()
}

#[test]
fn the_embedding_gather_decodes_every_weight_type() {
    let Some(gpus) = gpus() else { return };
    let types: Vec<GgmlType> =
        ares_core::ggml_type::ALL_TYPES.into_iter().filter(|t| t.is_weight()).collect();
    assert_eq!(types.len(), 28);
    for device in gpus {
        let gpu = Gpu::new(device);
        for &ty in &types {
            // Rows of three blocks (of 32 values for the float types, whose blocks are one
            // value), made of the blocks of the fixture in order.
            let n_embd = if ty.block_values() == 1 { 32 } else { 3 * ty.block_values() as usize };
            let row_bytes = ty.bytes_for(n_embd as u64).unwrap() as usize;
            let blocks = golden_blocks(ty);
            let n_vocab = blocks.len() / row_bytes;
            assert!(n_vocab >= 8, "{ty}: the fixture has {} rows of {n_embd} values", n_vocab);
            let table = &blocks[..n_vocab * row_bytes];
            // Every row once, backwards, then a repeated row, the first one and two tokens
            // past the table.
            let mut tokens: Vec<u32> = (0..n_vocab as u32).rev().collect();
            tokens.extend([3, 0, n_vocab as u32, 77 + n_vocab as u32]);
            let segments =
                [Segment { row0: 0, n_rows: tokens.len() as u32, slot: 0, pos0: 0, direct: false }];
            let record = StepRecord::new(&segments, &tokens, None, &[]).unwrap();
            let (step, table_buf) = (gpu.step(&record), gpu.bytes(table));
            let out = gpu.empty((tokens.len() * n_embd * 4) as u64);
            unsafe {
                gpu.layers
                    .embed_gather(
                        &gpu.stream,
                        step.ptr(),
                        record.dims(),
                        &EmbedRows {
                            ty,
                            n_vocab: n_vocab as u32,
                            n_embd: n_embd as u32,
                            kept: KeptRows {
                                rows: table_buf.ptr(),
                                tokens: DevicePtr::NULL,
                                count: n_vocab as u32,
                            },
                            staged: DevicePtr::NULL,
                        },
                        out.ptr(),
                    )
                    .unwrap();
            }
            let got = gpu.read(&out, tokens.len() * n_embd);
            for (r, &token) in tokens.iter().enumerate() {
                let mut want = vec![0.0f32; n_embd];
                if (token as usize) < n_vocab {
                    let src = &table[token as usize * row_bytes..][..row_bytes];
                    ares_quant::dequantize(ty, src, &mut want).unwrap();
                }
                // The kernel may fuse a multiply and an add where the reference does not: a
                // tolerance of the largest finite value of the row. Infinities (the largest
                // MXFP4 scales overflow) and the NaNs of the float types must be the same.
                let scale =
                    want.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
                for (i, (g, w)) in got[r * n_embd..][..n_embd].iter().zip(&want).enumerate() {
                    let ok = if w.is_finite() {
                        (g - w).abs() <= 2e-6 * scale
                    } else {
                        g == w || (g.is_nan() && w.is_nan())
                    };
                    assert!(
                        ok,
                        "{ty} row {r} (token {token}) value {i}: {g} for {w}, scale {scale}"
                    );
                }
            }
        }
        // A type that holds no weights is refused by name.
        let record = StepRecord::new(&[], &[], None, &[]).unwrap();
        let buf = gpu.bytes(&[0u8; 64]);
        for ty in [GgmlType::Q8_K, GgmlType::I8, GgmlType::F64] {
            let err = unsafe {
                gpu.layers.embed_gather(
                    &gpu.stream,
                    buf.ptr(),
                    record.dims(),
                    &EmbedRows {
                        ty,
                        n_vocab: 1,
                        n_embd: 256,
                        kept: KeptRows::NONE,
                        staged: buf.ptr(),
                    },
                    buf.ptr(),
                )
            };
            assert!(err.unwrap_err().to_string().contains(ty.name()), "{ty}");
        }
    }
}

/// A table of `n` rows of three Q8_0 blocks with scales of one, so that no value is a NaN.
fn q8_table(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut table: Vec<u8> = (0..n * 102).map(|_| (rng.unit() * 120.0) as i8 as u8).collect();
    assert_eq!(GgmlType::Q8_0.block_bytes(), 34);
    for block in table.as_chunks_mut::<34>().0 {
        block[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    }
    table
}

#[test]
fn the_embedding_gather_takes_a_row_from_the_rows_the_device_keeps_or_the_staged_one() {
    let Some(gpus) = gpus() else { return };
    let ty = GgmlType::Q8_0;
    let (n_vocab, n_embd) = (40usize, 96usize);
    let row_bytes = ty.bytes_for(n_embd as u64).unwrap() as usize;
    assert_eq!(row_bytes, 102);
    let mut rng = Rng(11);
    // The table, and the rows the host stages, which differ from it so that the source of
    // a row shows in its values.
    let table = q8_table(&mut rng, n_vocab);
    let other = q8_table(&mut rng, n_vocab);
    let decode = |rows: &[u8], row: usize| {
        let mut values = vec![0.0f32; n_embd];
        ares_quant::dequantize(ty, &rows[row * row_bytes..][..row_bytes], &mut values).unwrap();
        values
    };
    let zeros = vec![0.0f32; n_embd];
    // The device keeps the rows of the tokens 3, 4, 17, 18 and 30 (as a list), or of the
    // tokens 0 to 4 (as a prefix).
    let listed: Vec<u32> = vec![3, 4, 17, 18, 30];
    let listed_rows: Vec<u8> = listed
        .iter()
        .flat_map(|&t| table[t as usize * row_bytes..][..row_bytes].to_vec())
        .collect();
    let prefix = 5;
    // Tokens kept by both, by one, by neither, and past the table.
    let tokens: Vec<u32> = vec![17, 5, 30, 3, 39, 0, 4, 40, 41, u32::MAX, 18, 2];
    let n = tokens.len();
    for device in gpus {
        let gpu = Gpu::new(device);
        let segments = [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: 0, direct: false }];
        let record = StepRecord::new(&segments, &tokens, None, &[]).unwrap();
        let step = gpu.step(&record);
        // Row `r` of the step is staged from `other` for the token it has; a kept token's
        // staged row is the wrong one, which must not be read.
        let staged: Vec<u8> = tokens
            .iter()
            .flat_map(|&t| {
                let row = (t as usize + 1) % n_vocab;
                other[row * row_bytes..][..row_bytes].to_vec()
            })
            .collect();
        let (listed_buf, tokens_buf) = (
            gpu.bytes(&listed_rows),
            gpu.bytes(&listed.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()),
        );
        let prefix_buf = gpu.bytes(&table[..prefix * row_bytes]);
        let staged_buf = gpu.bytes(&staged);
        let run = |kept: KeptRows, staged: DevicePtr| {
            let out = gpu.empty((n * n_embd * 4) as u64);
            let source =
                EmbedRows { ty, n_vocab: n_vocab as u32, n_embd: n_embd as u32, kept, staged };
            unsafe {
                gpu.layers
                    .embed_gather(&gpu.stream, step.ptr(), record.dims(), &source, out.ptr())
                    .unwrap();
            }
            gpu.read(&out, n * n_embd)
        };
        let list = KeptRows { rows: listed_buf.ptr(), tokens: tokens_buf.ptr(), count: 5 };
        let first = KeptRows { rows: prefix_buf.ptr(), tokens: DevicePtr::NULL, count: 5 };
        let cases: [(&str, KeptRows, DevicePtr); 4] = [
            ("a list and staged rows", list, staged_buf.ptr()),
            ("a prefix and staged rows", first, staged_buf.ptr()),
            ("staged rows only", KeptRows::NONE, staged_buf.ptr()),
            ("a list only", list, DevicePtr::NULL),
        ];
        for (what, kept, staged_ptr) in cases {
            let got = run(kept, staged_ptr);
            for (r, &t) in tokens.iter().enumerate() {
                let in_list = listed.iter().position(|&l| l == t);
                let kept_row = match what {
                    "a prefix and staged rows" => (t as usize) < prefix,
                    "staged rows only" => false,
                    _ => in_list.is_some(),
                };
                let want = if t as usize >= n_vocab {
                    zeros.clone()
                } else if kept_row {
                    match what {
                        "a prefix and staged rows" => decode(&table, t as usize),
                        _ => decode(&listed_rows, in_list.unwrap()),
                    }
                } else if staged_ptr != DevicePtr::NULL {
                    decode(&staged, r)
                } else {
                    zeros.clone()
                };
                assert_eq!(
                    &got[r * n_embd..][..n_embd],
                    want.as_slice(),
                    "{what}: row {r} (token {t})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------

fn norm_reference(
    x: &[f32],
    weight: Option<&[f32]>,
    w: usize,
    eps: f32,
    kind: NormKind,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(w) {
        let ssq: f64 = row.iter().map(|v| f64::from(*v).powi(2)).sum();
        let denom = match kind {
            NormKind::Rms => ssq / w as f64 + f64::from(eps),
            NormKind::L2 => ssq + f64::from(eps),
        };
        let scale = 1.0 / denom.sqrt();
        for (j, v) in row.iter().enumerate() {
            out.push(f64::from(*v) * scale * weight.map_or(1.0, |wt| f64::from(wt[j])));
        }
    }
    out
}

#[test]
fn head_norms_match_the_formulas() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x5EED_0002);
        for &width in &[32usize, 64, 128, 256, 1024] {
            for &rows in &[1usize, 7, 37] {
                for kind in [NormKind::Rms, NormKind::L2] {
                    for with_weight in [false, true] {
                        let x = random_floats(&mut rng, rows * width, 5.0);
                        let weight: Vec<f32> = (0..width).map(|_| 0.5 + rng.unit().abs()).collect();
                        let eps = 1e-6;
                        let want = norm_reference(
                            &x,
                            with_weight.then_some(&weight[..]),
                            width,
                            eps,
                            kind,
                        );
                        let (xb, wb) = (gpu.floats(&x), gpu.floats(&weight));
                        let yb = gpu.empty((x.len() * 4) as u64);
                        unsafe {
                            gpu.layers
                                .head_norm(
                                    &gpu.stream,
                                    kind,
                                    xb.ptr(),
                                    with_weight.then(|| wb.ptr()),
                                    yb.ptr(),
                                    rows as u32,
                                    width as u32,
                                    eps,
                                )
                                .unwrap();
                        }
                        let err = rel_max(&gpu.read(&yb, x.len()), &want);
                        assert!(
                            err < 3e-7,
                            "{kind:?} w={width} rows={rows} weight={with_weight}: {err}"
                        );
                        // In place gives the same.
                        unsafe {
                            gpu.layers
                                .head_norm(
                                    &gpu.stream,
                                    kind,
                                    xb.ptr(),
                                    with_weight.then(|| wb.ptr()),
                                    xb.ptr(),
                                    rows as u32,
                                    width as u32,
                                    eps,
                                )
                                .unwrap();
                        }
                        assert_eq!(gpu.read(&xb, x.len()), gpu.read(&yb, x.len()));
                    }
                }
            }
        }
        // Widths the kernel does not do are refused.
        let b = gpu.floats(&[0.0; 64]);
        assert!(
            unsafe {
                gpu.layers.head_norm(&gpu.stream, NormKind::Rms, b.ptr(), None, b.ptr(), 1, 6, 1e-6)
            }
            .is_err()
        );
        assert!(
            unsafe {
                gpu.layers.head_norm(
                    &gpu.stream,
                    NormKind::Rms,
                    b.ptr(),
                    None,
                    b.ptr(),
                    1,
                    2048,
                    1e-6,
                )
            }
            .is_err()
        );
    }
}

#[test]
fn the_gather_of_output_rows_copies_and_norms() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x5EED_0003);
        let (n_rows, d) = (11usize, 1024usize);
        let x = random_floats(&mut rng, n_rows * d, 3.0);
        let weight: Vec<f32> = (0..d).map(|_| 0.5 + rng.unit().abs()).collect();
        let out_rows = [10u32, 0, 4, 4, 7];
        let segments = [Segment { row0: 0, n_rows: 11, slot: 0, pos0: 0, direct: false }];
        let record = StepRecord::new(&segments, &vec![0; n_rows], None, &out_rows).unwrap();
        let (step, xb, wb) = (gpu.step(&record), gpu.floats(&x), gpu.floats(&weight));
        let yb = gpu.empty((out_rows.len() * d * 4) as u64);
        unsafe {
            gpu.layers
                .gather_rows(
                    &gpu.stream,
                    step.ptr(),
                    record.dims(),
                    xb.ptr(),
                    None,
                    yb.ptr(),
                    d as u32,
                    1e-6,
                )
                .unwrap();
        }
        let copied = gpu.read(&yb, out_rows.len() * d);
        for (i, &r) in out_rows.iter().enumerate() {
            assert_eq!(&copied[i * d..(i + 1) * d], &x[r as usize * d..(r as usize + 1) * d]);
        }
        unsafe {
            gpu.layers
                .gather_rows(
                    &gpu.stream,
                    step.ptr(),
                    record.dims(),
                    xb.ptr(),
                    Some(wb.ptr()),
                    yb.ptr(),
                    d as u32,
                    1e-6,
                )
                .unwrap();
        }
        let picked: Vec<f32> = out_rows
            .iter()
            .flat_map(|&r| x[r as usize * d..(r as usize + 1) * d].to_vec())
            .collect();
        let want = norm_reference(&picked, Some(&weight), d, 1e-6, NormKind::Rms);
        let err = rel_max(&gpu.read(&yb, out_rows.len() * d), &want);
        assert!(err < 3e-7, "{err}");
    }
}

// ---------------------------------------------------------------------------

/// The quantized-activation bytes the weight kernels expect for `values`, made by
/// `Kernels::quantize_activations` from the f32 copy that a fused kernel
/// wrote into its own output. Equal bytes mean the fused kernel's codes, scales
/// and sums are those of the separate kernel for the values it computed.
fn check_fused_layout(gpu: &Gpu, fused: &DeviceBuffer, n_values: usize, what: &str) -> Vec<f32> {
    let total = quantized_activation_bytes(n_values as u64) as usize;
    let fused_bytes = gpu.read_bytes(fused, total);
    let x_at = total - n_values * 4;
    let copy: Vec<f32> =
        fused_bytes[x_at..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let (xb, qb) = (gpu.floats(&copy), gpu.empty(total as u64));
    unsafe {
        gpu.kernels
            .quantize_activations(&gpu.stream, xb.ptr(), qb.ptr(), n_values as u32, 1)
            .unwrap();
    }
    let separate = gpu.read_bytes(&qb, total);
    assert_eq!(
        total as u64,
        Kernels::quantized_activation_bytes(n_values as u32, 1),
        "{what}: size"
    );
    let codes = n_values;
    assert_eq!(&fused_bytes[..codes], &separate[..codes], "{what}: int8 codes");
    let ds_at = codes.next_multiple_of(16);
    let ds_end = ds_at + n_values / 32 * 8;
    assert_eq!(&fused_bytes[ds_at..ds_end], &separate[ds_at..ds_end], "{what}: scales and sums");
    assert_eq!(&fused_bytes[x_at..], &separate[x_at..], "{what}: f32 copy");
    copy
}

#[test]
fn swiglu_with_quantization_is_the_separate_kernels() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x5EED_0004);
        for &n_values in &[32usize, 96, 4096, 17408 * 2] {
            let gate = random_floats(&mut rng, n_values, 6.0);
            let up = random_floats(&mut rng, n_values, 2.0);
            let (gb, ub) = (gpu.floats(&gate), gpu.floats(&up));
            let act = gpu.empty(quantized_activation_bytes(n_values as u64));
            unsafe {
                gpu.layers
                    .swiglu_quant(&gpu.stream, gb.ptr(), ub.ptr(), act.ptr(), n_values as u32)
                    .unwrap();
            }
            let copy = check_fused_layout(&gpu, &act, n_values, "swiglu");
            let want: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| {
                    let g = f64::from(*g);
                    g / (1.0 + (-g).exp()) * f64::from(*u)
                })
                .collect();
            let err = rel_max(&copy, &want);
            assert!(err < 3e-7, "swiglu n={n_values}: {err}");
        }
        let b = gpu.floats(&[0.0; 64]);
        assert!(
            unsafe { gpu.layers.swiglu_quant(&gpu.stream, b.ptr(), b.ptr(), b.ptr(), 40) }.is_err()
        );
    }
}

#[test]
fn the_gated_norm_with_quantization_matches_its_formula_and_the_layout() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x5EED_0005);
        for &(heads, width, rows) in
            &[(48usize, 128usize, 1usize), (16, 128, 7), (6, 64, 5), (4, 256, 3)]
        {
            for gate in [OutputGate::Silu, OutputGate::Sigmoid] {
                let n = rows * heads * width;
                let o = random_floats(&mut rng, n, 4.0);
                let z = random_floats(&mut rng, n, 5.0);
                let weight: Vec<f32> = (0..width).map(|_| 0.3 + rng.unit().abs()).collect();
                let (ob, zb, wb) = (gpu.floats(&o), gpu.floats(&z), gpu.floats(&weight));
                let act = gpu.empty(quantized_activation_bytes(n as u64));
                unsafe {
                    gpu.layers
                        .gated_norm_quant(
                            &gpu.stream,
                            gate,
                            ob.ptr(),
                            zb.ptr(),
                            wb.ptr(),
                            act.ptr(),
                            (rows * heads) as u32,
                            width as u32,
                            1e-6,
                        )
                        .unwrap();
                }
                let copy = check_fused_layout(&gpu, &act, n, "gated norm");
                let normed = norm_reference(&o, Some(&weight), width, 1e-6, NormKind::Rms);
                let want: Vec<f64> = normed
                    .iter()
                    .zip(&z)
                    .map(|(y, z)| {
                        let z = f64::from(*z);
                        let sigmoid = 1.0 / (1.0 + (-z).exp());
                        y * match gate {
                            OutputGate::Silu => z * sigmoid,
                            OutputGate::Sigmoid => sigmoid,
                        }
                    })
                    .collect();
                let err = rel_max(&copy, &want);
                assert!(err < 5e-7, "{gate:?} {heads}x{width} rows={rows}: {err}");
            }
        }
        let b = gpu.floats(&[0.0; 64]);
        assert!(
            unsafe {
                gpu.layers.gated_norm_quant(
                    &gpu.stream,
                    OutputGate::Silu,
                    b.ptr(),
                    b.ptr(),
                    b.ptr(),
                    b.ptr(),
                    1,
                    96,
                    1e-6,
                )
            }
            .is_err()
        );
    }
}

// ---------------------------------------------------------------------------

/// `(cos, sin)` of pair `p` of a row with positions `pos` (time, height, width, extra).
fn rope_pair(pos: [i32; 4], p: usize, rope: &RopeParams) -> (f64, f64) {
    let total: usize = rope.sections.iter().map(|s| *s as usize).sum();
    let sector = p % total;
    let s = rope.sections.map(|s| s as usize);
    let stream = if sector.is_multiple_of(3) && sector < 3 * s[0] {
        0
    } else if sector % 3 == 1 && sector < 3 * s[1] {
        1
    } else if sector % 3 == 2 && sector < 3 * s[2] {
        2
    } else {
        3
    };
    let freq = f64::from(rope.freq_base).powf(-2.0 * p as f64 / f64::from(rope.dim));
    let angle = f64::from(pos[stream]) * freq;
    (angle.cos(), angle.sin())
}

#[test]
fn the_rope_table_and_the_attention_preparation_match_the_formulas() {
    let Some(gpus) = gpus() else { return };
    let rope = RopeParams { dim: 64, freq_base: 1e7, sections: [11, 11, 10, 0] };
    let shape = AttentionShape { n_head: 6, n_kv_head: 2, head_dim: 256 };
    let (nh, nkv, hd) = (shape.n_head as usize, shape.n_kv_head as usize, shape.head_dim as usize);
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x5EED_0006);
        // Two sequences: one at position 0, one deep in its history, and a row with
        // three different position kinds (an image token) to exercise the sections.
        let segments = [
            Segment { row0: 0, n_rows: 3, slot: 0, pos0: 0, direct: true },
            Segment { row0: 3, n_rows: 2, slot: 1, pos0: 262_100, direct: false },
        ];
        let tokens = [0u32; 6];
        let mut positions = vec![[0i32; 4]; 6];
        for (t, p) in positions.iter_mut().enumerate().take(3) {
            *p = [t as i32; 4];
            p[3] = 0;
        }
        positions[3] = [262_100, 262_100, 262_100, 0];
        positions[4] = [262_101, 262_101, 262_101, 0];
        positions[5] = [100, 7, 35, 3];
        let segments5 = [
            segments[0],
            segments[1],
            Segment { row0: 5, n_rows: 1, slot: 2, pos0: 100, direct: false },
        ];
        let record = StepRecord::new(&segments5, &tokens, Some(&positions), &[]).unwrap();
        let n = 6usize;
        let step = gpu.step(&record);
        let half = rope.dim as usize / 2;
        let table = gpu.empty((n * half * 8) as u64);
        unsafe {
            gpu.layers
                .rope_table(&gpu.stream, step.ptr(), record.dims(), table.ptr(), rope)
                .unwrap();
        }
        let got = gpu.read(&table, n * half * 2);
        for r in 0..n {
            for p in 0..half {
                let (c, s) = rope_pair(positions[r], p, &rope);
                let (gc, gs) =
                    (f64::from(got[(r * half + p) * 2]), f64::from(got[(r * half + p) * 2 + 1]));
                assert!(
                    (gc - c).abs() < 1e-7 && (gs - s).abs() < 1e-7,
                    "row {r} pair {p}: ({gc}, {gs}) vs ({c}, {s})"
                );
            }
        }

        let qg = random_floats(&mut rng, n * nh * 2 * hd, 3.0);
        let k = random_floats(&mut rng, n * nkv * hd, 3.0);
        let q_norm: Vec<f32> = (0..hd).map(|_| 0.5 + rng.unit().abs()).collect();
        let k_norm: Vec<f32> = (0..hd).map(|_| 0.5 + rng.unit().abs()).collect();
        let (qgb, kb, qnb, knb) =
            (gpu.floats(&qg), gpu.floats(&k), gpu.floats(&q_norm), gpu.floats(&k_norm));
        for with_gate in [true, false] {
            let q_out = gpu.empty((n * nh * hd * 4) as u64);
            let gate_out = gpu.empty((n * nh * hd * 4) as u64);
            let k_out = gpu.empty((n * nkv * hd * 4) as u64);
            unsafe {
                gpu.layers
                    .attn_prep(
                        &gpu.stream,
                        step.ptr(),
                        record.dims(),
                        shape,
                        rope.dim,
                        1e-6,
                        qgb.ptr(),
                        kb.ptr(),
                        qnb.ptr(),
                        knb.ptr(),
                        table.ptr(),
                        q_out.ptr(),
                        with_gate.then(|| gate_out.ptr()),
                        k_out.ptr(),
                    )
                    .unwrap();
            }
            let mut want_q = vec![0.0f64; n * nh * hd];
            let mut want_g = vec![0.0f64; n * nh * hd];
            let mut want_k = vec![0.0f64; n * nkv * hd];
            let rotate = |row: usize, head: &mut [f64]| {
                let src = head.to_vec();
                for p in 0..half {
                    let (c, s) = rope_pair(positions[row], p, &rope);
                    head[p] = src[p] * c - src[p + half] * s;
                    head[p + half] = src[p] * s + src[p + half] * c;
                }
            };
            for r in 0..n {
                for h in 0..nh {
                    let src = &qg[(r * nh + h) * 2 * hd..][..2 * hd];
                    let mut head =
                        norm_reference(&src[..hd], Some(&q_norm), hd, 1e-6, NormKind::Rms);
                    rotate(r, &mut head);
                    want_q[(r * nh + h) * hd..][..hd].copy_from_slice(&head);
                    for j in 0..hd {
                        want_g[(r * nh + h) * hd + j] = f64::from(src[hd + j]);
                    }
                }
                for h in 0..nkv {
                    let src = &k[(r * nkv + h) * hd..][..hd];
                    let mut head = norm_reference(src, Some(&k_norm), hd, 1e-6, NormKind::Rms);
                    rotate(r, &mut head);
                    want_k[(r * nkv + h) * hd..][..hd].copy_from_slice(&head);
                }
            }
            let (eq, ek) = (
                rel_max(&gpu.read(&q_out, n * nh * hd), &want_q),
                rel_max(&gpu.read(&k_out, n * nkv * hd), &want_k),
            );
            assert!(eq < 5e-7 && ek < 5e-7, "q {eq}, k {ek}");
            if with_gate {
                assert_eq!(rel_max(&gpu.read(&gate_out, n * nh * hd), &want_g), 0.0);
            } else {
                assert!(
                    gpu.read(&gate_out, 4).iter().all(|v| v.is_nan()),
                    "the gate must be left alone"
                );
            }
        }
    }
}
