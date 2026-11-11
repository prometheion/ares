//! Tests of the layer kernels (`shaders/layer_*.comp`, `src/layers.rs`) that need
//! nothing but the quantization reference: the step record, the embedding
//! gather against `ares_quant::dequantize`, the norms, the fused quantization
//! (byte for byte against `Kernels::quantize_activations`), the rotary table and
//! the attention preparation against plain f64 formulas, and the merges of a tensor
//! split. The recurrent kernels (convolution, delta rule, commit) are held to
//! `ares_model`'s reference in `crates/model/tests/layers_vulkan.rs`, which needs that
//! crate.
//!
//! This file is the twin of `crates/cuda/tests/layers.rs`: the same references and the
//! same tolerances, so that the two backends are held to one standard.
//!
//! Like the other GPU tests they run only with `ARES_TEST_VULKAN=1`:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test layers
//! ```

// The tests launch kernels on buffers they allocated a few lines earlier, with sizes taken from
// the same shapes; a comment on each of the launches would repeat that.
#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_vulkan::{
    AttentionShape, Device, DeviceBuffer, Kernels, LayerKernels, MergeNorm, NormKind, OutputGate,
    RopeParams, Segment, StepRecord, Stream, Wire, device_count, partial_wire_bytes,
    quantized_activation_bytes, quantized_activation_copy_offset,
};

/// The devices every test shares, opened once and never closed: the process leaves through
/// [`leave`].
static DEVICES: OnceLock<Option<&'static [Device]>> = OnceLock::new();

/// Ends the process at once when it exits, with the status it exits with (glibc's `on_exit`
/// hands it over): no exit handler of the driver runs and no device is destroyed. The NVIDIA
/// driver (615.78) compiles pipelines on threads of its own, and it crashes there when a
/// device is destroyed or its exit handlers run while one of them still works: seen as a
/// segmentation fault with devices that were never destroyed, and as an abort ("corrupted
/// size vs. prev_size" in `libnvidia-gpucomp`) with devices dropped by a handler at exit
/// after a long run that had made hundreds of pipelines. The kernel frees what the process
/// held.
extern "C" fn leave(status: i32, _: *mut std::ffi::c_void) {
    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }
    // SAFETY: `_exit` takes any status and does not return.
    unsafe { _exit(status) }
}

/// Registers [`leave`] for the exit of the process.
fn leave_at_exit() {
    unsafe extern "C" {
        fn on_exit(
            callback: extern "C" fn(i32, *mut std::ffi::c_void),
            arg: *mut std::ffi::c_void,
        ) -> i32;
    }
    // SAFETY: registering a function that ends the process; it uses no argument.
    unsafe { on_exit(leave, std::ptr::null_mut()) };
}

fn gpus() -> Option<&'static [Device]> {
    *DEVICES.get_or_init(|| {
        if std::env::var_os("ARES_TEST_VULKAN").is_none() {
            eprintln!("skipped: set ARES_TEST_VULKAN=1 to run the tests that open GPUs");
            return None;
        }
        let count = device_count().expect("the Vulkan loader");
        assert!(count > 0, "ARES_TEST_VULKAN is set but the loader reports no device");
        let opened: Vec<Device> =
            (0..count).map(|i| Device::open(i).expect("open device")).collect();
        leave_at_exit();
        Some(Box::leak(opened.into_boxed_slice()) as &'static [Device])
    })
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
        let kernels = Kernels::load(device).unwrap();
        Self {
            device: device.clone(),
            stream: Stream::new(device).unwrap(),
            layers: LayerKernels::load(&kernels).unwrap(),
            kernels,
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

/// Prints the largest error a test saw on a device (`--nocapture`), as `ERR name value`.
fn report(gpu: &Gpu, name: &str, worst: f64) {
    eprintln!("ERR {name} [{}] {worst:.3e}", gpu.device.info().name);
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
        let mut seen = 0.0f64;
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
                        table_buf.ptr(),
                        ty,
                        n_vocab as u32,
                        n_embd as u32,
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
                        seen = seen.max(f64::from((g - w).abs() / scale.max(f32::MIN_POSITIVE)));
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
        report(&gpu, "embed_gather", seen);
        // A type that holds no weights is refused by name.
        let record = StepRecord::new(&[], &[], None, &[]).unwrap();
        let buf = gpu.bytes(&[0u8; 64]);
        for ty in [GgmlType::Q8_K, GgmlType::I8, GgmlType::F64] {
            let err = unsafe {
                gpu.layers.embed_gather(
                    &gpu.stream,
                    buf.ptr(),
                    record.dims(),
                    buf.ptr(),
                    ty,
                    1,
                    256,
                    buf.ptr(),
                )
            };
            assert!(err.unwrap_err().to_string().contains(ty.name()), "{ty}");
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
        let mut seen = 0.0f64;
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
                        seen = seen.max(err);
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
        report(&gpu, "head_norm", seen);
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
        report(&gpu, "gather_rows", err);
        assert!(err < 3e-7, "{err}");
    }
}

// ---------------------------------------------------------------------------

/// Checks the output of a fused kernel over `n_values` values against
/// `Kernels::quantize_activations` run on the f32 values the fused kernel kept behind its
/// codes and pairs, and returns those values. Equal bytes mean the fused kernel's codes,
/// scales and sums are those of the separate kernel for the values it computed, so the
/// weight kernels cannot tell the two apart.
fn check_fused_layout(gpu: &Gpu, fused: &DeviceBuffer, n_values: usize, what: &str) -> Vec<f32> {
    let total = quantized_activation_bytes(n_values as u64) as usize;
    let fused_bytes = gpu.read_bytes(fused, total);
    let x_at = quantized_activation_copy_offset(n_values as u64) as usize;
    assert_eq!(total, x_at + n_values * 4, "{what}: size");
    let copy: Vec<f32> =
        fused_bytes[x_at..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    // What the weight kernels read: the codes and the pairs.
    let read = Kernels::quantized_activation_bytes(n_values as u32, 1) as usize;
    assert_eq!(read.next_multiple_of(16), x_at, "{what}: the copy lies behind the pairs");
    let (xb, qb) = (gpu.floats(&copy), gpu.empty(read as u64));
    unsafe {
        gpu.kernels
            .quantize_activations(&gpu.stream, xb.ptr(), qb.ptr(), n_values as u32, 1)
            .unwrap();
    }
    let separate = gpu.read_bytes(&qb, read);
    let codes = n_values;
    assert_eq!(&fused_bytes[..codes], &separate[..codes], "{what}: int8 codes");
    let ds_at = codes.next_multiple_of(16);
    let ds_end = ds_at + n_values / 32 * 8;
    assert_eq!(ds_end, read, "{what}: pairs end");
    assert_eq!(&fused_bytes[ds_at..ds_end], &separate[ds_at..ds_end], "{what}: scales and sums");
    copy
}

#[test]
fn swiglu_with_quantization_is_the_separate_kernels() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x5EED_0004);
        let mut seen = 0.0f64;
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
            seen = seen.max(err);
            assert!(err < 3e-7, "swiglu n={n_values}: {err}");
        }
        report(&gpu, "swiglu_quant", seen);
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
        let mut seen = 0.0f64;
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
                seen = seen.max(err);
                assert!(err < 5e-7, "{gate:?} {heads}x{width} rows={rows}: {err}");
            }
        }
        report(&gpu, "gated_norm_quant", seen);
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
        let mut seen = 0.0f64;
        for r in 0..n {
            for p in 0..half {
                let (c, s) = rope_pair(positions[r], p, &rope);
                let (gc, gs) =
                    (f64::from(got[(r * half + p) * 2]), f64::from(got[(r * half + p) * 2 + 1]));
                seen = seen.max((gc - c).abs()).max((gs - s).abs());
                assert!(
                    (gc - c).abs() < 1e-7 && (gs - s).abs() < 1e-7,
                    "row {r} pair {p}: ({gc}, {gs}) vs ({c}, {s})"
                );
            }
        }

        report(&gpu, "rope_table(absolute)", seen);

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
            report(&gpu, "attn_prep", eq.max(ek));
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

// ---------------------------------------------------------------------------

/// A row as a reader of its wire form gets it: rounded to f16 after the division by the
/// power of two that keeps it in range, or as it is in the wide form. Returns the factor too.
fn through_the_wire(row: &[f32], form: Wire) -> (Vec<f32>, f32) {
    if form == Wire::F32 {
        return (row.to_vec(), 1.0);
    }
    let m = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let e = ((m.to_bits() >> 23) & 0xff) as i32 - 126;
    let k = (e - 15).max(0);
    let (down, up) = (2f32.powi(-k), 2f32.powi(k));
    (
        row.iter().map(|v| ares_quant::f16_to_f32(ares_quant::f32_to_f16(v * down)) * up).collect(),
        up,
    )
}

/// The merge of a tensor split on one device standing in for two: both partials are put
/// into wire form and merged once as the first device and once as the second. The wire
/// bytes and the sum must be the host's bit for bit (so every device holds the same
/// residual stream), the norm the formula's, and the quantized form the separate kernel's.
#[test]
fn the_merges_of_a_tensor_split_give_the_same_sum_and_its_norm() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        let mut rng = Rng(0x5EED_0007);
        let mut seen = 0.0f64;
        for &(n_rows, d) in &[(5usize, 1024usize), (3, 96), (4, 5120), (1, 32)] {
            for form in [Wire::F16, Wire::F32] {
                // Row 1 is large enough to overflow f16 and row 2 is tiny.
                let rows = |rng: &mut Rng, scale: f32| -> Vec<f32> {
                    (0..n_rows * d)
                        .map(|i| {
                            let factor = match i / d {
                                1 => 3.0e5,
                                2 => 1.0e-6,
                                _ => 1.0,
                            };
                            rng.unit() * scale * factor
                        })
                        .collect()
                };
                let (pa, pb) = (rows(&mut rng, 2.0), rows(&mut rng, 3.0));
                let x0 = rows(&mut rng, 5.0);
                let weight: Vec<f32> = (0..d).map(|_| 0.5 + rng.unit().abs()).collect();
                let wire_bytes = partial_wire_bytes(form, n_rows as u32, d as u32);
                let (pab, pbb, wb) = (gpu.floats(&pa), gpu.floats(&pb), gpu.floats(&weight));
                let (wa, wbuf) = (gpu.empty(wire_bytes), gpu.empty(wire_bytes));
                unsafe {
                    for (p, wire) in [(&pab, &wa), (&pbb, &wbuf)] {
                        gpu.layers
                            .partial_to_wire(
                                &gpu.stream,
                                form,
                                p.ptr(),
                                wire.ptr(),
                                n_rows as u32,
                                d as u32,
                            )
                            .unwrap();
                    }
                }
                // The wire bytes are the host's.
                let mut want = vec![0.0f32; n_rows * d];
                for (p, wire) in [(&pa, &wa), (&pb, &wbuf)] {
                    let got = gpu.read_bytes(wire, wire_bytes as usize);
                    let values_bytes = match form {
                        Wire::F16 => (n_rows * d * 2).next_multiple_of(4),
                        Wire::F32 => n_rows * d * 4,
                    };
                    for r in 0..n_rows {
                        let (seen, up) = through_the_wire(&p[r * d..(r + 1) * d], form);
                        let factor = f32::from_le_bytes(
                            got[values_bytes + 4 * r..][..4].try_into().unwrap(),
                        );
                        assert_eq!(factor, up, "{form:?} {n_rows}x{d}: factor of row {r}");
                        for (j, s) in seen.iter().enumerate() {
                            let on_wire = match form {
                                Wire::F16 => {
                                    let at = (r * d + j) * 2;
                                    ares_quant::f16_to_f32(u16::from_le_bytes([
                                        got[at],
                                        got[at + 1],
                                    ])) * factor
                                }
                                Wire::F32 => f32::from_le_bytes(
                                    got[(r * d + j) * 4..][..4].try_into().unwrap(),
                                ),
                            };
                            assert_eq!(
                                on_wire.to_bits(),
                                s.to_bits(),
                                "{form:?} {n_rows}x{d}: row {r} value {j} on the wire"
                            );
                            want[r * d + j] += s;
                        }
                    }
                }
                // x + (a + b), a first: the order of the devices.
                let sum: Vec<f32> = x0.iter().zip(&want).map(|(x, s)| x + s).collect();
                let sum64: Vec<f64> = sum.iter().map(|v| f64::from(*v)).collect();
                let normed = norm_reference(&sum, Some(&weight), d, 1e-6, NormKind::Rms);
                for own_rank in [0u32, 1] {
                    let (own, peer) = if own_rank == 0 { (&wa, &wbuf) } else { (&wbuf, &wa) };
                    for mode in 0..3 {
                        let xb = gpu.floats(&x0);
                        let xn = gpu.empty((n_rows * d * 4) as u64);
                        let act = gpu.empty(quantized_activation_bytes((n_rows * d) as u64));
                        let norm = match mode {
                            0 => MergeNorm::None,
                            1 => MergeNorm::Rows { weight: wb.ptr(), out: xn.ptr() },
                            _ => MergeNorm::Quantized { weight: wb.ptr(), act: act.ptr() },
                        };
                        unsafe {
                            gpu.layers
                                .merge_wire(
                                    &gpu.stream,
                                    form,
                                    xb.ptr(),
                                    own.ptr(),
                                    peer.ptr(),
                                    wire_bytes.next_multiple_of(16),
                                    1,
                                    own_rank,
                                    n_rows as u32,
                                    d as u32,
                                    norm,
                                    1e-6,
                                )
                                .unwrap();
                        }
                        let what = format!("{form:?} {n_rows}x{d} rank {own_rank} mode {mode}");
                        let got = gpu.read(&xb, n_rows * d);
                        for (i, (g, w)) in got.iter().zip(&sum).enumerate() {
                            assert_eq!(g.to_bits(), w.to_bits(), "{what}: value {i} of the sum");
                        }
                        assert_eq!(rel_max(&got, &sum64), 0.0);
                        match mode {
                            1 => {
                                // Row by row: the rows differ by eleven orders of magnitude.
                                for r in 0..n_rows {
                                    let err = rel_max(
                                        &gpu.read(&xn, n_rows * d)[r * d..(r + 1) * d],
                                        &normed[r * d..(r + 1) * d],
                                    );
                                    seen = seen.max(err);
                                    assert!(err < 3e-7, "{what}: norm of row {r}: {err}");
                                }
                            }
                            2 => {
                                let copy = check_fused_layout(&gpu, &act, n_rows * d, &what);
                                for r in 0..n_rows {
                                    let err = rel_max(
                                        &copy[r * d..(r + 1) * d],
                                        &normed[r * d..(r + 1) * d],
                                    );
                                    seen = seen.max(err);
                                    assert!(err < 3e-7, "{what}: quantized norm of row {r}: {err}");
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        report(&gpu, "merge_wire(norm)", seen);
        // Rows that are not whole groups of 32 values are refused.
        let b = gpu.floats(&[0.0; 64]);
        assert!(
            unsafe { gpu.layers.partial_to_wire(&gpu.stream, Wire::F16, b.ptr(), b.ptr(), 1, 40) }
                .is_err()
        );
    }
}
