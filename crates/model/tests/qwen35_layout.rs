//! The layouts of the weight matrices in the device implementation of
//! `qwen35`: the fused kernels that write f16 activations for matrices in
//! their device layout, the model in either layout against the CPU reference,
//! and the text with the model's own drafts.
//!
//! - `ARES_TEST_CUDA=1` alone runs the kernels' test (`ARES_TEST_DEVICE`
//!   picks the GPU, default 0).
//! - With `ARES_TEST_QWEN35_MODEL` (the 0.8B file) and
//!   `ARES_TEST_QWEN35_CAPTURE` (the capture directory of `tools/capture`)
//!   the model is run with every matrix in the file's layout, with every
//!   matrix that can have it in its device layout, and with the measured
//!   choice.
//! - With two GPUs and `ARES_TEST_QWEN35_27B` the text with the model's own
//!   drafts is compared with the text without, with the default products
//!   (`ARES_TEST_DRAFT_VOCAB` names a draft vocabulary).
//!
//! Lines that start with `ERR` or `RES` are the measurements.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

use ares_api::{
    Engine, Event, EventSink, GenerateRequest, Message, Prompt, RequestId, Role, Sampling,
};
use ares_cuda::{
    ActivationForm, Device, DeviceBuffer, Kernels, LayerKernels, MergeNorm, OutputGate, Precision,
    Segment, Stream, Wait, Wire, device_count, partial_wire_bytes, quantized_activation_bytes,
};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Harness, Layouts, Options, Qwen35, Spread};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig};

fn gpu() -> Option<&'static Device> {
    static DEVICE: OnceLock<Option<Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            if std::env::var_os("ARES_TEST_CUDA").is_none() {
                eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
                return None;
            }
            let ordinal =
                std::env::var("ARES_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            Some(Device::open(ordinal, Wait::Block).expect("open device"))
        })
        .as_ref()
}

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set {name} to run this test");
            None
        }
    }
}

/// Values in (-`scale`, `scale`) from a fixed sequence.
fn noise(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

fn upload(device: &Device, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = DeviceBuffer::new(device, bytes.len().max(64) as u64).unwrap();
    buffer.write(0, &bytes).unwrap();
    buffer
}

fn download(buffer: &DeviceBuffer, bytes: u64) -> Vec<u8> {
    let mut raw = vec![0u8; bytes as usize];
    buffer.read(0, &mut raw).unwrap();
    raw
}

/// The bytes that count of `n` rows of `n_in` f16 activations: the values of
/// every row (the rows lie further apart than their values) and the rows'
/// factors.
fn f16_rows(raw: &[u8], n_in: u32, n: u32) -> Vec<u8> {
    let pitch = ((((n_in as usize * 2).div_ceil(128)) + 1) | 1) * 128;
    let mut out = Vec::new();
    for row in 0..n as usize {
        out.extend_from_slice(&raw[row * pitch..row * pitch + n_in as usize * 2]);
    }
    let factors = (pitch * n as usize).next_multiple_of(16);
    out.extend_from_slice(&raw[factors..factors + 4 * n as usize]);
    out
}

/// The fused kernels write, byte for byte, the f16 activations that a
/// conversion of the values they compute writes: the gated norm and SwiGLU
/// against the f32 copy their quantizing twins keep, the merge of a tensor
/// split against its own normalised rows, and the merge that writes both
/// forms against the one that quantizes.
#[test]
fn the_fused_kernels_write_the_f16_activations_of_a_conversion() {
    let Some(device) = gpu() else { return };
    let kernels = Kernels::load(device).unwrap();
    let layers = LayerKernels::load(device).unwrap();
    let stream = Stream::new(device, 0).unwrap();
    let f16_bytes = |n_in: u32, n: u32| Kernels::activation_bytes(ActivationForm::F16, n_in, n);
    // Where the f32 copy of the values is in the quantized layout.
    let copy_at = |values: u64| quantized_activation_bytes(values) - values * 4;
    let convert = |rows: ares_cuda::DevicePtr, n_in: u32, n: u32| -> Vec<u8> {
        let out = DeviceBuffer::new(device, f16_bytes(n_in, n)).unwrap();
        // SAFETY: `rows` holds `n` rows of `n_in` f32 values, `out` their f16 form.
        unsafe {
            kernels
                .prepare_activations(&stream, ActivationForm::F16, rows, out.ptr(), n_in, n)
                .unwrap();
        }
        stream.synchronize().unwrap();
        f16_rows(&download(&out, f16_bytes(n_in, n)), n_in, n)
    };

    // The gated norm: heads of every width, rows of one head to more heads
    // than a block has warps, both gates.
    for (case, &(n, heads, width)) in
        [(1u32, 6u32, 128u32), (3, 40, 128), (9, 2, 64), (2, 1, 256), (5, 33, 64)]
            .iter()
            .enumerate()
    {
        let n_in = heads * width;
        let values = u64::from(n) * u64::from(n_in);
        let o = upload(device, &noise(11 + case as u64, values as usize, 3.0));
        let z = upload(device, &noise(23 + case as u64, values as usize, 4.0));
        let weight = upload(device, &noise(31 + case as u64, width as usize, 2.0));
        let act = DeviceBuffer::new(device, quantized_activation_bytes(values)).unwrap();
        let got = DeviceBuffer::new(device, f16_bytes(n_in, n)).unwrap();
        for gate in [OutputGate::Silu, OutputGate::Sigmoid] {
            // SAFETY: the buffers hold `n * heads` items of `width` values and
            // their activations in either form.
            unsafe {
                layers
                    .gated_norm_quant(
                        &stream,
                        gate,
                        o.ptr(),
                        z.ptr(),
                        weight.ptr(),
                        act.ptr(),
                        n * heads,
                        width,
                        1e-6,
                    )
                    .unwrap();
                layers
                    .gated_norm_f16(
                        &stream,
                        gate,
                        o.ptr(),
                        z.ptr(),
                        weight.ptr(),
                        got.ptr(),
                        n,
                        heads,
                        width,
                        1e-6,
                    )
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let want = convert(act.ptr().offset(copy_at(values)), n_in, n);
            let got = f16_rows(&download(&got, f16_bytes(n_in, n)), n_in, n);
            assert!(got == want, "gated norm, {n} rows of {heads} heads of {width}, {gate:?}");
        }
    }

    // SwiGLU: rows up to the width of a large feed-forward part.
    for (case, &(n, n_in)) in [(1u32, 3584u32), (4, 17408), (7, 32), (2, 10880)].iter().enumerate()
    {
        let values = u64::from(n) * u64::from(n_in);
        let gate = upload(device, &noise(41 + case as u64, values as usize, 5.0));
        let up = upload(device, &noise(53 + case as u64, values as usize, 3.0));
        let act = DeviceBuffer::new(device, quantized_activation_bytes(values)).unwrap();
        let got = DeviceBuffer::new(device, f16_bytes(n_in, n)).unwrap();
        // SAFETY: the buffers hold `n` rows of `n_in` values and their
        // activations in either form.
        unsafe {
            layers.swiglu_quant(&stream, gate.ptr(), up.ptr(), act.ptr(), n * n_in).unwrap();
            layers.swiglu_f16(&stream, gate.ptr(), up.ptr(), got.ptr(), n, n_in).unwrap();
        }
        stream.synchronize().unwrap();
        let want = convert(act.ptr().offset(copy_at(values)), n_in, n);
        let got = f16_rows(&download(&got, f16_bytes(n_in, n)), n_in, n);
        assert!(got == want, "swiglu, {n} rows of {n_in}");
    }

    // The merge, with this device's partial alone: every norm leaves the same
    // residual stream, the f16 activations are those of the normalised rows,
    // and the quantized ones of the merge that writes both are those of the
    // merge that quantizes.
    for wire in [Wire::F16, Wire::F32] {
        for (case, &(n, d)) in [(1u32, 5120u32), (4, 1024), (8, 2560), (3, 96)].iter().enumerate() {
            let values = u64::from(n) * u64::from(d);
            let p = upload(device, &noise(61 + case as u64, values as usize, 2.0));
            let x0 = noise(67 + case as u64, values as usize, 30.0);
            let weight = upload(device, &noise(71 + case as u64, d as usize, 2.0));
            let parcel = DeviceBuffer::new(device, partial_wire_bytes(wire, n, d)).unwrap();
            let xn = DeviceBuffer::new(device, values * 4).unwrap();
            let quantized = DeviceBuffer::new(device, quantized_activation_bytes(values)).unwrap();
            let both_q = DeviceBuffer::new(device, quantized_activation_bytes(values)).unwrap();
            let (half, both_h) = (
                DeviceBuffer::new(device, f16_bytes(d, n)).unwrap(),
                DeviceBuffer::new(device, f16_bytes(d, n)).unwrap(),
            );
            let (w, q, h) = (weight.ptr(), quantized.ptr(), half.ptr());
            let norms = [
                MergeNorm::Rows { weight: w, out: xn.ptr() },
                MergeNorm::Quantized { weight: w, act: q },
                MergeNorm::F16 { weight: w, act: h },
                MergeNorm::Both { weight: w, quantized: both_q.ptr(), f16: both_h.ptr() },
            ];
            let mut streams = Vec::new();
            for norm in norms {
                let x = upload(device, &x0);
                // SAFETY: `p` and `x` hold `n` rows of `d` values, the parcel
                // their wire form, and the norm's buffers what it writes.
                unsafe {
                    layers.partial_to_wire(&stream, wire, p.ptr(), parcel.ptr(), n, d).unwrap();
                    layers
                        .merge_wire(
                            &stream,
                            wire,
                            x.ptr(),
                            parcel.ptr(),
                            parcel.ptr(),
                            16,
                            0,
                            0,
                            n,
                            d,
                            norm,
                            1e-6,
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
                streams.push(download(&x, values * 4));
            }
            assert!(streams.iter().all(|s| *s == streams[0]), "merge {wire:?} {n} x {d}: x");
            let want = convert(xn.ptr(), d, n);
            let bytes = f16_bytes(d, n);
            assert!(
                f16_rows(&download(&half, bytes), d, n) == want,
                "merge {wire:?} {n} x {d}: f16"
            );
            assert!(
                f16_rows(&download(&both_h, bytes), d, n) == want,
                "merge {wire:?} {n} x {d}: f16 of both"
            );
            let bytes = quantized_activation_bytes(values);
            assert!(
                download(&both_q, bytes) == download(&quantized, bytes),
                "merge {wire:?} {n} x {d}: quantized of both"
            );
        }
    }
}

/// The logits of a head that lies on the device in parts: the rows of each
/// part put in their place.
#[test]
fn rows_of_parts_are_put_in_their_place() {
    let Some(device) = gpu() else { return };
    let layers = LayerKernels::load(device).unwrap();
    let stream = Stream::new(device, 0).unwrap();
    let (n, widths) = (5u32, [777u32, 1234, 3]);
    let total: u32 = widths.iter().sum();
    let dst = upload(device, &vec![-1.0f32; (n * total) as usize]);
    let mut want = vec![0.0f32; (n * total) as usize];
    let mut at = 0;
    for (p, width) in widths.into_iter().enumerate() {
        let src = noise(81 + p as u64, (n * width) as usize, 9.0);
        for r in 0..n as usize {
            let row = &src[r * width as usize..(r + 1) * width as usize];
            want[r * total as usize + at as usize..][..width as usize].copy_from_slice(row);
        }
        let src = upload(device, &src);
        // SAFETY: `src` holds `n` rows of `width` values, `dst` `n` rows of
        // `total`.
        unsafe { layers.place_rows(&stream, src.ptr(), dst.ptr(), n, width, total, at).unwrap() };
        stream.synchronize().unwrap();
        at += width;
    }
    let got = download(&dst, u64::from(n * total) * 4);
    let want: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert!(got == want, "rows in their place");
    // A part that does not fit the row is refused.
    let src = upload(device, &[0.0; 8]);
    // SAFETY: refused before anything is launched.
    let refused =
        unsafe { layers.place_rows(&stream, src.ptr(), dst.ptr(), 1, 8, total, total - 4) };
    assert!(refused.is_err());
}

/// What the fused kernels cost with one row, by the form they write: printed,
/// not asserted (`RES fused` lines, microseconds per launch, the fastest of
/// five runs of sixteen launches).
#[test]
fn the_fused_kernels_by_form() {
    let Some(device) = gpu() else { return };
    let layers = LayerKernels::load(device).unwrap();
    let stream = Stream::new(device, 0).unwrap();
    let (start, end) =
        (ares_cuda::Event::timed(device).unwrap(), ares_cuda::Event::timed(device).unwrap());
    let time = |launch: &dyn Fn()| -> f32 {
        let mut best = f32::INFINITY;
        for _ in 0..6 {
            start.record(&stream).unwrap();
            for _ in 0..16 {
                launch();
            }
            end.record(&stream).unwrap();
            end.synchronize().unwrap();
            best = best.min(end.elapsed_ms_since(&start).unwrap() / 16.0 * 1e3);
        }
        best
    };
    let name = device.info().name.clone();
    for n in [1u32, 4] {
        // The merge of a tensor split at the width of Qwen3.8-27B.
        let d = 5120u32;
        let values = u64::from(n) * u64::from(d);
        let p = upload(device, &noise(3, values as usize, 2.0));
        let x = upload(device, &noise(5, values as usize, 30.0));
        let weight = upload(device, &noise(7, d as usize, 2.0));
        let parcel = DeviceBuffer::new(device, partial_wire_bytes(Wire::F16, n, d)).unwrap();
        let xn = DeviceBuffer::new(device, values * 4).unwrap();
        let act_q = DeviceBuffer::new(device, quantized_activation_bytes(values)).unwrap();
        let act_h = DeviceBuffer::new(device, Kernels::activation_bytes(ActivationForm::F16, d, n))
            .unwrap();
        // SAFETY: the buffers hold `n` rows of `d` values in every form.
        unsafe { layers.partial_to_wire(&stream, Wire::F16, p.ptr(), parcel.ptr(), n, d).unwrap() };
        let w = weight.ptr();
        let norms = [
            ("rows", MergeNorm::Rows { weight: w, out: xn.ptr() }),
            ("8-bit", MergeNorm::Quantized { weight: w, act: act_q.ptr() }),
            ("f16", MergeNorm::F16 { weight: w, act: act_h.ptr() }),
            ("both", MergeNorm::Both { weight: w, quantized: act_q.ptr(), f16: act_h.ptr() }),
        ];
        let mut line = format!("RES fused {name} merge of {n} row(s) of {d}:");
        for (what, norm) in norms {
            let us = time(&|| {
                // SAFETY: as above.
                unsafe {
                    layers
                        .merge_wire(
                            &stream,
                            Wire::F16,
                            x.ptr(),
                            parcel.ptr(),
                            parcel.ptr(),
                            16,
                            0,
                            0,
                            n,
                            d,
                            norm,
                            1e-6,
                        )
                        .unwrap();
                }
            });
            line += &format!(" {what} {us:.2} us,");
        }
        eprintln!("{line}");

        // The gated norm and SwiGLU at the widths of a device's slice.
        for (heads, inner) in [(30u32, 10880u32), (18, 6528)] {
            let n_in = heads * 128;
            let o = upload(device, &noise(11, (n * n_in) as usize, 3.0));
            let z = upload(device, &noise(13, (n * n_in) as usize, 4.0));
            let gate = upload(device, &noise(17, (n * inner) as usize, 5.0));
            let up = upload(device, &noise(19, (n * inner) as usize, 3.0));
            let bytes = |n_in: u32| {
                quantized_activation_bytes(u64::from(n) * u64::from(n_in))
                    .max(Kernels::activation_bytes(ActivationForm::F16, n_in, n))
            };
            let act = DeviceBuffer::new(device, bytes(n_in.max(inner))).unwrap();
            let silu = OutputGate::Silu;
            // SAFETY: the buffers hold `n` rows of the widths and either form.
            let times = unsafe {
                [
                    time(&|| {
                        layers
                            .gated_norm_quant(
                                &stream,
                                silu,
                                o.ptr(),
                                z.ptr(),
                                w,
                                act.ptr(),
                                n * heads,
                                128,
                                1e-6,
                            )
                            .unwrap()
                    }),
                    time(&|| {
                        layers
                            .gated_norm_f16(
                                &stream,
                                silu,
                                o.ptr(),
                                z.ptr(),
                                w,
                                act.ptr(),
                                n,
                                heads,
                                128,
                                1e-6,
                            )
                            .unwrap()
                    }),
                    time(&|| {
                        layers
                            .swiglu_quant(&stream, gate.ptr(), up.ptr(), act.ptr(), n * inner)
                            .unwrap()
                    }),
                    time(&|| {
                        layers
                            .swiglu_f16(&stream, gate.ptr(), up.ptr(), act.ptr(), n, inner)
                            .unwrap()
                    }),
                ]
            };
            eprintln!(
                "RES fused {name} {n} row(s): gated norm of {heads} heads 8-bit {:.2} us, f16 {:.2} \
                 us; swiglu of {inner} 8-bit {:.2} us, f16 {:.2} us",
                times[0], times[1], times[2], times[3]
            );
        }
    }
}

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

/// `|got - want| / |want|` over all values (L2).
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        d += (f64::from(*g) - f64::from(*w)).powi(2);
        n += f64::from(*w).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

/// The largest value of `x`, its index, and its lead over the second.
fn argmax(x: &[f32]) -> (u32, f32) {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    let second = x
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best)
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    (best as u32, x[best] - second)
}

fn direct(n: usize, pos0: usize) -> [Segment; 1] {
    [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: pos0 as i32, direct: true }]
}

/// The 0.8B with every matrix in the file's layout, with every matrix that
/// can have it in its device layout, and with the measured choice, against
/// the CPU reference: the error of every block alone, and the most likely
/// token of every prompt row. The device layout is the exact arithmetic of
/// f16 operands at every number of rows, so its fast products are held to the
/// limits of the exact ones for a block; the file's layout keeps the limits
/// of 8-bit activations.
#[test]
fn the_0_8b_in_either_layout_matches_the_reference() {
    let (Some(device), Some(path), Some(capture)) =
        (gpu(), env_path("ARES_TEST_QWEN35_MODEL"), env_path("ARES_TEST_QWEN35_CAPTURE"))
    else {
        return;
    };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    let policies = [
        ("file", Layouts::File, Precision::Fast),
        ("device", Layouts::Device, Precision::Fast),
        ("measured", Layouts::default(), Precision::Fast),
        ("exact", Layouts::default(), Precision::Exact),
    ];
    let options = |layouts, precision| Options {
        precision,
        layouts,
        max_rows: 256,
        logit_rows: 16,
        window_rows: 16,
        ..Options::default()
    };

    // Every block alone, with the hidden state and the state the reference
    // has there: one row, a window of four, and a prompt's rows.
    let ids = read_ids(&capture.join("german/base/prompt.tokens"));
    let history = 40usize;
    let mut state = SequenceState::new(&c);
    reference.forward(&mut state, &ids[..history], LogitRows::None).unwrap();
    for (name, layouts, precision) in policies {
        let mut harness =
            Harness::open(device, &c, &w, &options(layouts, precision), 1, 512).unwrap();
        let report = harness.part().layouts().clone();
        eprintln!(
            "RES layouts {name}: {} matrices in the file's layout, {} in the device layout, {} \
             shapes probed in {:.0} ms, tables {:.0} ms",
            report.matrices[0],
            report.matrices[1],
            report.probes.len(),
            report.probes_time.as_secs_f64() * 1e3,
            report.tables.as_secs_f64() * 1e3
        );
        match layouts {
            Layouts::File => assert_eq!(report.matrices[1], 0),
            // The embedding table lies in host memory, and the 0.8B's head, which
            // is the same tensor, is a head like any other.
            Layouts::Device => assert_eq!(report.matrices[0], 0),
            Layouts::Measured { .. } => {}
        }
        for n in [1usize, 4, 21] {
            let tokens = &ids[history..history + n];
            let window = reference.window(&state, tokens, LogitRows::None).unwrap();
            let (mut worst_r, mut worst_a) = (0.0f64, 0.0f64);
            for layer in 0..c.n_layer {
                let x_in: Vec<f32> = if layer == 0 {
                    tokens
                        .iter()
                        .flat_map(|t| {
                            let mut row = vec![0.0f32; c.n_embd];
                            w.token_embd.decode_row(*t as usize, &mut row);
                            row
                        })
                        .collect()
                } else {
                    window.hidden[layer - 1].clone()
                };
                let want = reference.block(layer, &state, &x_in);
                if c.is_attention(layer) {
                    let (k, v) = state.kv(layer).unwrap();
                    harness.write_kv(layer, 0, history as u32, k, v).unwrap();
                } else {
                    harness
                        .write_recurrent_state(
                            layer,
                            0,
                            state.conv(layer).unwrap(),
                            state.delta(layer).unwrap(),
                        )
                        .unwrap();
                }
                let got = harness.block(layer, &direct(n, history), &x_in).unwrap();
                let e = rel_l2(&got, &want);
                if c.is_attention(layer) {
                    worst_a = worst_a.max(e);
                } else {
                    worst_r = worst_r.max(e);
                }
            }
            eprintln!("ERR block {name} n={n}: recurrent {worst_r:.2e} attention {worst_a:.2e}");
            // The limits of `qwen35_device.rs`: exact products, or 8-bit
            // activations. K and V are q8 here, which an attention block
            // carries in either case.
            let limit = match (layouts, precision) {
                (Layouts::Device, _) | (_, Precision::Exact) => 5e-3,
                _ => 5e-2,
            };
            assert!(worst_r < limit, "{name} n={n}: recurrent block {worst_r}");
            assert!(worst_a < 5e-2, "{name} n={n}: attention block {worst_a}");
        }
    }

    // The whole forward pass over the captured prompts, rows in steps of 16:
    // the most likely token of every row against the reference's.
    let mut totals = [(0usize, 0usize); 4];
    for language in ["english", "german", "code"] {
        let ids = read_ids(&capture.join(language).join("base/prompt.tokens"));
        let n = ids.len();
        let mut state = SequenceState::new(&c);
        let window = reference.forward(&mut state, &ids, LogitRows::All).unwrap();
        let want: Vec<(u32, f32)> = (0..n).map(|r| argmax(window.logits_of(r).unwrap())).collect();
        let last = window.logits_of(n - 1).unwrap().to_vec();
        for (p, (name, layouts, precision)) in policies.into_iter().enumerate() {
            let mut harness =
                Harness::open(device, &c, &w, &options(layouts, precision), 1, 512).unwrap();
            let (mut picks, mut got_last) = (Vec::new(), Vec::new());
            for (chunk, tokens) in ids.chunks(16).enumerate() {
                let rows: Vec<u32> = (0..tokens.len() as u32).collect();
                let logits =
                    harness.step(&direct(tokens.len(), chunk * 16), tokens, &rows).unwrap();
                let vocab = logits.len() / tokens.len();
                picks.extend(logits.chunks_exact(vocab).map(|row| argmax(row).0));
                got_last = logits[logits.len() - vocab..].to_vec();
            }
            let agree = picks.iter().zip(&want).filter(|(p, w)| **p == w.0).count();
            let widest_miss = picks
                .iter()
                .zip(&want)
                .filter(|(p, w)| **p != w.0)
                .map(|(_, w)| w.1)
                .fold(0.0f32, f32::max);
            eprintln!(
                "RES top1 {language} {name}: {agree}/{n} with the reference (widest missed \
                 margin {widest_miss:.3}); last row's logits {:.2e} from the reference",
                rel_l2(&got_last, &last)
            );
            totals[p].0 += agree;
            totals[p].1 += n;
            assert!(agree * 100 >= n * 90, "{language} {name}: top-1 {agree}/{n}");
        }
    }
    for ((name, ..), (agree, n)) in policies.iter().zip(totals) {
        eprintln!("RES top1 all prompts {name}: {agree}/{n} with the reference");
    }
}

fn ask(runtime: &Runtime, id: u64, question: &str, sampling: Sampling, max_tokens: u32) -> String {
    let (tx, rx) = mpsc::channel();
    let text = Arc::new(Mutex::new(String::new()));
    let sink = {
        let text = text.clone();
        EventSink::new(move |event| {
            match event {
                Event::Text(piece) => text.lock().unwrap().push_str(&piece),
                Event::Logprobs(_) => {}
                Event::Done { .. } => {
                    let _ = tx.send(Ok(()));
                }
                Event::Error(error) => {
                    let _ = tx.send(Err(error.to_string()));
                }
            }
            true
        })
    };
    runtime
        .submit(
            GenerateRequest {
                model: "q".into(),
                prompt: Prompt::Chat {
                    messages: vec![Message::new(Role::User, question)],
                    tools: None,
                    variables: Vec::new(),
                    continue_final: false,
                },
                sampling,
                max_tokens: Some(max_tokens),
                stop: Vec::new(),
                draft: true,
                structure: None,
                priority: 0,
                id: RequestId(id),
            },
            sink,
        )
        .unwrap();
    rx.recv().unwrap().unwrap();
    std::mem::take(&mut *text.lock().unwrap())
}

/// The answers of the 27B over both devices to `questions`, each with the
/// most probable token, with at most `drafts` of the model's own drafts per
/// window, and how many matrices each device holds in `[file, device]`
/// layout.
fn answers(
    model: &Path,
    vocabulary: Option<&Arc<[u32]>>,
    layouts: Layouts,
    drafts: u32,
    questions: &[&str],
) -> (Vec<String>, Vec<[u32; 2]>) {
    let options = Options {
        precision: Precision::Fast,
        layouts,
        max_rows: 128,
        logit_rows: 16,
        window_rows: if drafts > 0 { drafts + 1 } else { 0 },
        draft_vocabulary: vocabulary.cloned(),
        ..Options::default()
    };
    let model = Arc::new(
        Qwen35::new(
            Gguf::open(model).unwrap(),
            "q",
            &Spread::Tensor(vec![0.625, 0.375]),
            options,
            1024,
        )
        .unwrap(),
    );
    let config = RuntimeConfig {
        pages: 20,
        cells_per_page: 128,
        max_slots: 2,
        window_rows: drafts + 1,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig::new(&config, &*model);
    let devices =
        vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
    let runtime = Runtime::start_gpu(model.clone(), devices, config, gpu).unwrap();
    let greedy = Sampling { temperature: Some(0.0), ..Sampling::default() };
    let texts = questions
        .iter()
        .enumerate()
        .map(|(i, question)| ask(&runtime, i as u64 + 1, question, greedy.clone(), 64))
        .collect();
    runtime.shutdown();
    (texts, model.reports().iter().map(|r| r.layouts.matrices).collect())
}

/// With the default products, is the text with the model's own drafts the
/// text without them? With exact products it is (`qwen35_mtp_runtime.rs`).
/// The few-row kernels compute a row as the same bits whatever the rows beside
/// it and however they launch, but the rows of one sequence in a step still
/// change each other's logits a little through the recurrent or the attention
/// layers (2e-4 of the logit on the 27B with every matrix in its device
/// layout, 1e-3 in the file's layout: see "Speculation" in
/// `docs/ARCHITECTURE.md`), so the texts agree as far as that stays below the
/// margin of their picks, which it did for the two questions here with
/// every matrix in its device layout. The measured choice puts other matrices in
/// the file's layout in every load (more with an MTP block), so its two
/// models are not one model, and in the file's layout the difference above is
/// four to seven times larger: its texts are reported, not compared.
#[test]
fn the_text_with_drafts_and_fast_products() {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return;
    }
    let Some(model) = env_path("ARES_TEST_QWEN35_27B") else { return };
    if device_count().unwrap() < 2 {
        eprintln!("skipped: the 27B needs two devices");
        return;
    }
    let vocabulary: Option<Arc<[u32]>> = std::env::var_os("ARES_TEST_DRAFT_VOCAB").map(|path| {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.parse().unwrap())
            .collect()
    });
    let questions = [
        "Erkl\u{e4}re in drei S\u{e4}tzen, warum der Himmel blau ist.",
        "Write a Python function that returns the n-th Fibonacci number.",
    ];
    for (name, layouts) in [("measured", Layouts::default()), ("device", Layouts::Device)] {
        let (plain, plain_layouts) = answers(&model, vocabulary.as_ref(), layouts, 0, &questions);
        let (drafted, drafted_layouts) =
            answers(&model, vocabulary.as_ref(), layouts, 3, &questions);
        eprintln!(
            "RES drafts fast {name}: matrices [file, device] per device without drafts \
             {plain_layouts:?}, with drafts {drafted_layouts:?}"
        );
        // Every matrix in the device layout is one assignment in both models.
        let comparable = matches!(layouts, Layouts::Device);
        for (i, (a, b)) in plain.iter().zip(&drafted).enumerate() {
            let same = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            eprintln!(
                "RES drafts fast {name} answer {i}: {same} of {} characters equal the text \
                 without drafts{}",
                a.chars().count(),
                if a == b { " (the same text)" } else { "" }
            );
            assert!(!a.is_empty() && !b.is_empty(), "{name}: an answer is empty");
            if comparable {
                assert_eq!(a, b, "{name}: answer {i} with drafts is not the text without");
            }
        }
    }
}
