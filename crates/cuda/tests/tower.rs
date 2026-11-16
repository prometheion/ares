//! Tests of the image tower's kernels (`TowerKernels`) against host
//! computations: the f16 activations its norm and GELU write are the form of
//! `prepare_prefill_activations`, byte for byte; the residual, the placing of
//! parts and the positions are exact; the rotation and the attention are within
//! the rounding of their f16 operands.
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test tower -- --test-threads=1 --nocapture
//! ```

use std::sync::OnceLock;

use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, HeadShape, Kernels, PositionGrid, PrefillPath, Stream,
    TowerKernels, Wait, device_count,
};
use ares_quant::{f16_to_f32, f32_to_f16};

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

/// A small deterministic generator, so failures reproduce.
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

    fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

fn upload(device: &Device, values: &[f32]) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, (values.len().max(1) * 4) as u64).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    buffer.write(0, &bytes).unwrap();
    buffer
}

fn download(buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buffer.read(0, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn bytes_of(buffer: &DeviceBuffer, n: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n];
    buffer.read(0, &mut bytes).unwrap();
    bytes
}

/// The f32 values of `n_rows` rows of `n_in` f16 activations (the form of
/// `prepare_prefill_activations` on the f16 path): each value times its row's
/// factor, which is exact.
fn decode_act(bytes: &[u8], n_in: usize, n_rows: usize) -> Vec<f32> {
    let line = Kernels::ACTIVATION_LINE as usize;
    let pitch = (((n_in * 2).div_ceil(line) + 1) | 1) * line / 2;
    let factors = (n_rows * pitch * 2).next_multiple_of(16);
    let mut out = Vec::with_capacity(n_in * n_rows);
    for r in 0..n_rows {
        let at = factors + 4 * r;
        let factor = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        for i in 0..n_in {
            let at = 2 * (r * pitch + i);
            out.push(f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]])) * factor);
        }
    }
    out
}

/// Encodes f32 rows with `prepare_prefill_activations` and returns the bytes.
fn convert(
    kernels: &Kernels,
    stream: &Stream,
    values: &[f32],
    n_in: usize,
    n_rows: usize,
) -> Vec<u8> {
    let device = kernels.device();
    let x = upload(device, values);
    let bytes = Kernels::prefill_activation_bytes(PrefillPath::F16, n_in as u32, n_rows as u32);
    let out = DeviceBuffer::new(device, bytes).unwrap();
    // SAFETY: the buffers hold the rows and the prepared bytes.
    unsafe {
        kernels
            .prepare_prefill_activations(
                stream,
                PrefillPath::F16,
                x.ptr(),
                out.ptr(),
                n_in as u32,
                n_rows as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    bytes_of(&out, bytes as usize)
}

/// The layer norm and the GELU write the form of `prepare_prefill_activations`:
/// what they write, decoded and encoded again by it, is the same bytes; the
/// values are the host's within the f16 rounding of the row; padding is zero.
#[test]
fn norm_and_gelu_write_the_f16_form_of_the_tile_kernels() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let kernels = Kernels::load(device).unwrap();
        let tower = TowerKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut rng = Rng(7);
        let (n, n_rows) = (1152usize, 9usize);
        for group in [1usize, 4] {
            let rows_in = n_rows * group;
            // Values around an offset, some rows with a large one: the residual's look.
            let mut x = rng.values(rows_in * n, 3.0);
            for (r, row) in x.chunks_mut(n).enumerate() {
                let offset = if r % 3 == 0 { 400.0 } else { 0.5 };
                row.iter_mut().for_each(|v| *v += offset);
                row[r % n] = 6000.0;
            }
            let weight = rng.values(n, 2.0);
            let bias = rng.values(n, 0.5);
            let width = n * group;
            let bytes =
                Kernels::prefill_activation_bytes(PrefillPath::F16, width as u32, n_rows as u32);
            let out = DeviceBuffer::new(device, bytes).unwrap();
            let (xb, wb, bb) = (upload(device, &x), upload(device, &weight), upload(device, &bias));
            // SAFETY: the buffers have the sizes the launch reads and writes.
            unsafe {
                tower
                    .norm_f16(
                        &stream,
                        xb.ptr(),
                        wb.ptr(),
                        bb.ptr(),
                        out.ptr(),
                        n as u32,
                        group as u32,
                        n_rows as u32,
                        1e-6,
                    )
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let ours = bytes_of(&out, bytes as usize);
            let decoded = decode_act(&ours, width, n_rows);
            assert_eq!(convert(&kernels, &stream, &decoded, width, n_rows), ours, "group {group}");
            // Against the host's norm in f64.
            for (r, (row, out)) in x.chunks(n).zip(decoded.chunks(n)).enumerate() {
                let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
                let var =
                    row.iter().map(|v| (f64::from(*v) - mean).powi(2)).sum::<f64>() / n as f64;
                let scale = 1.0 / (var + 1e-6).sqrt();
                let expect: Vec<f64> = (0..n)
                    .map(|i| {
                        (f64::from(row[i]) - mean) * scale * f64::from(weight[i])
                            + f64::from(bias[i])
                    })
                    .collect();
                let top = expect.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for (a, b) in out.iter().zip(&expect) {
                    assert!((f64::from(*a) - b).abs() <= top * 1e-3, "row {r}: {a} against {b}");
                }
            }
        }

        // GELU: rows of 4304 values padded to 4320.
        let (n, pad) = (4304usize, 4320usize);
        let y = rng.values(n_rows * n, 30.0);
        let bias = rng.values(n, 5.0);
        let bytes = Kernels::prefill_activation_bytes(PrefillPath::F16, pad as u32, n_rows as u32);
        let out = DeviceBuffer::new(device, bytes).unwrap();
        let (yb, bb) = (upload(device, &y), upload(device, &bias));
        // SAFETY: as above.
        unsafe {
            tower
                .gelu_f16(
                    &stream,
                    yb.ptr(),
                    bb.ptr(),
                    out.ptr(),
                    n as u32,
                    pad as u32,
                    n_rows as u32,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let ours = bytes_of(&out, bytes as usize);
        let decoded = decode_act(&ours, pad, n_rows);
        assert_eq!(convert(&kernels, &stream, &decoded, pad, n_rows), ours, "GELU");
        for (row, out) in y.chunks(n).zip(decoded.chunks(pad)) {
            assert!(out[n..].iter().all(|v| *v == 0.0), "padding");
            let gelu = |x: f64| {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * x * (1.0 + 0.044715 * x * x))
                            .tanh())
            };
            let expect: Vec<f64> =
                row.iter().zip(&bias).map(|(v, b)| gelu(f64::from(*v + *b))).collect();
            let top = expect.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (a, b) in out.iter().zip(&expect) {
                assert!((f64::from(*a) - b).abs() <= top * 1e-3, "GELU {a} against {b}");
            }
        }
    }
}

/// The residual, the placing of a part's columns and the positions are the
/// host's operations in `f32`, bit for bit.
#[test]
fn residual_place_and_positions_are_exact() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let tower = TowerKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut rng = Rng(11);
        let (n, rows) = (1152usize, 37usize);
        let x = rng.values(rows * n, 5000.0);
        let y = rng.values(rows * n, 3.0);
        let bias = rng.values(n, 1.0);
        let (xb, yb, bb) = (upload(device, &x), upload(device, &y), upload(device, &bias));
        // SAFETY: the buffers hold the rows.
        unsafe {
            tower.residual(&stream, xb.ptr(), yb.ptr(), bb.ptr(), n as u32, rows as u32).unwrap()
        };
        stream.synchronize().unwrap();
        let ours = download(&xb, rows * n);
        for (i, v) in ours.iter().enumerate() {
            assert_eq!(v.to_bits(), (x[i] + (y[i] + bias[i % n])).to_bits(), "residual {i}");
        }

        let (width, dst_width, at) = (300usize, 1000usize, 640usize);
        let src = rng.values(rows * width, 2.0);
        let part_bias = rng.values(width, 1.0);
        let dst = DeviceBuffer::new(device, (rows * dst_width * 4) as u64).unwrap();
        dst.write(0, &vec![0u8; rows * dst_width * 4]).unwrap();
        let (sb, pb) = (upload(device, &src), upload(device, &part_bias));
        // SAFETY: as above.
        unsafe {
            tower
                .place(
                    &stream,
                    sb.ptr(),
                    dst.ptr(),
                    pb.ptr(),
                    rows as u32,
                    width as u32,
                    dst_width as u32,
                    at as u32,
                )
                .unwrap();
            tower
                .place(
                    &stream,
                    sb.ptr(),
                    dst.ptr(),
                    DevicePtr::NULL,
                    rows as u32,
                    width as u32,
                    dst_width as u32,
                    0,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let placed = download(&dst, rows * dst_width);
        for r in 0..rows {
            for i in 0..width {
                let s = src[r * width + i];
                assert_eq!(placed[r * dst_width + at + i].to_bits(), (s + part_bias[i]).to_bits());
                assert_eq!(placed[r * dst_width + i].to_bits(), s.to_bits());
            }
            assert!(placed[r * dst_width + width..r * dst_width + at].iter().all(|v| *v == 0.0));
        }

        // Positions: a table of 6 x 6 cells of 8 values, grids larger, smaller and equal.
        let (side, n) = (6usize, 8usize);
        let table = rng.values(side * side * n, 4.0);
        let patch_bias = rng.values(n, 1.0);
        let (tb, pb) = (upload(device, &table), upload(device, &patch_bias));
        for (pw, ph) in [(6usize, 6usize), (10, 4), (2, 14), (4, 2)] {
            let factor = |m: usize| (m - 1) as f32 / (side - 1) as f32;
            let patches = pw * ph;
            let x = rng.values(patches * n, 2.0);
            let xb = upload(device, &x);
            let grid = PositionGrid {
                n: n as u32,
                side: side as u32,
                pw: pw as u32,
                ph: ph as u32,
                sfx: factor(pw),
                sfy: factor(ph),
            };
            // SAFETY: as above.
            unsafe { tower.positions(&stream, xb.ptr(), pb.ptr(), tb.ptr(), grid).unwrap() };
            stream.synchronize().unwrap();
            let ours = download(&xb, patches * n);
            let nx = pw / 2;
            for p in 0..patches {
                let (py, px) = (2 * (p / 4 / nx) + p % 4 / 2, 2 * (p / 4 % nx) + p % 2);
                let cell = |y: usize, x: usize, i: usize| table[(y * side + x) * n + i];
                let around = |i: usize, sf: f32| {
                    let pos = i as f32 / sf;
                    let lo = (pos.floor() as i64).clamp(0, side as i64 - 1);
                    let hi = (pos.floor() as i64 + 1).clamp(0, side as i64 - 1);
                    (lo as usize, hi as usize, (pos - lo as f32).clamp(0.0, 1.0))
                };
                for i in 0..n {
                    let position = if pw == side && ph == side {
                        cell(py, px, i)
                    } else {
                        let ((y0, y1, dy), (x0, x1, dx)) =
                            (around(py, grid.sfy), around(px, grid.sfx));
                        cell(y0, x0, i) * (1.0 - dx) * (1.0 - dy)
                            + cell(y0, x1, i) * dx * (1.0 - dy)
                            + cell(y1, x0, i) * (1.0 - dx) * dy
                            + cell(y1, x1, i) * dx * dy
                    };
                    let expect = (x[p * n + i] + patch_bias[i]) + position;
                    assert_eq!(
                        ours[p * n + i].to_bits(),
                        expect.to_bits(),
                        "{pw} x {ph}: patch {p} value {i}"
                    );
                }
            }
        }
    }
}

/// Query, key and value from a product's rows: bias, rotation by the patch's
/// row and column, f16 rows per head with zeros up to the pitch; then the
/// attention over them against the host's in `f64`, for patch counts that
/// leave partial tiles of queries and keys.
#[test]
fn rotation_and_attention_match_the_host() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let tower = TowerKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut rng = Rng(23);
        let (n_head, d) = (3usize, 72usize);
        let width = n_head * d;
        let step = 10000f32.powf(-2.0 / (d / 2) as f32);
        for (nx, ny) in [(1usize, 1usize), (5, 5), (12, 9)] {
            let patches = 4 * nx * ny;
            let shape =
                HeadShape { n_head: n_head as u32, head_dim: d as u32, n_patches: patches as u32 };
            let qkv = rng.values(patches * 3 * width, 4.0);
            let bias = rng.values(3 * width, 1.0);
            let heads = DeviceBuffer::new(device, shape.bytes()).unwrap();
            let (qb, bb) = (upload(device, &qkv), upload(device, &bias));
            // In two calls, as the groups of a block do.
            let half = (patches / 2) as u32;
            // SAFETY: the buffers hold the rows and the heads of `shape`.
            unsafe {
                tower
                    .rotate_heads(
                        &stream,
                        qb.ptr(),
                        bb.ptr(),
                        heads.ptr(),
                        shape,
                        0,
                        half,
                        nx as u32,
                        step,
                    )
                    .unwrap();
                let rest = qb.ptr().offset(u64::from(half) * 3 * width as u64 * 4);
                tower
                    .rotate_heads(
                        &stream,
                        rest,
                        bb.ptr(),
                        heads.ptr(),
                        shape,
                        half,
                        patches as u32 - half,
                        nx as u32,
                        step,
                    )
                    .unwrap();
            }
            let out = DeviceBuffer::new(device, (patches * width * 4) as u64).unwrap();
            let scale = 1.0 / (d as f32).sqrt();
            // SAFETY: as above.
            unsafe { tower.attention(&stream, heads.ptr(), out.ptr(), shape, scale).unwrap() };
            stream.synchronize().unwrap();
            let pitch = shape.pitch() as usize;
            let raw = bytes_of(&heads, shape.bytes() as usize);
            let half_at = |part: usize, head: usize, p: usize, k: usize| {
                let at = 2 * (((part * n_head + head) * patches + p) * pitch + k);
                f16_to_f32(u16::from_le_bytes([raw[at], raw[at + 1]]))
            };
            // The rows per head, against the host's rotation rounded to f16.
            let mut host = vec![vec![vec![0f32; patches * d]; n_head]; 3];
            let pairs = d / 2;
            for (part, heads) in host.iter_mut().enumerate() {
                for (head, rows) in heads.iter_mut().enumerate() {
                    let at = |k: usize| part * width + head * d + k;
                    for p in 0..patches {
                        let (py, px) = (2 * (p / 4 / nx) + p % 4 / 2, 2 * (p / 4 % nx) + p % 2);
                        let angle = |j: usize| {
                            let half = pairs / 2;
                            let mut theta = if j < half { py as f32 } else { px as f32 };
                            for _ in 0..(if j < half { j } else { j - half }) {
                                theta *= step;
                            }
                            theta
                        };
                        let v = |k: usize| qkv[p * 3 * width + at(k)] + bias[at(k)];
                        for k in 0..pitch {
                            let ours = half_at(part, head, p, k);
                            if k >= d {
                                assert_eq!(ours, 0.0, "padding");
                                continue;
                            }
                            let value = if part == 2 {
                                f64::from(v(k))
                            } else {
                                let theta = angle(k % pairs);
                                let (c, s) = (f64::from(theta.cos()), f64::from(theta.sin()));
                                if k < pairs {
                                    f64::from(v(k)) * c - f64::from(v(k + pairs)) * s
                                } else {
                                    f64::from(v(k)) * c + f64::from(v(k - pairs)) * s
                                }
                            };
                            let expect = f16_to_f32(f32_to_f16(value as f32));
                            let ulp = f16_to_f32(f32_to_f16(expect.abs()) + 1) - expect.abs();
                            assert!(
                                (ours - expect).abs() <= ulp,
                                "{nx} x {ny}: part {part} head {head} patch {p} value {k}: {ours} \
                                 against {expect}"
                            );
                            rows[p * d + k] = ours;
                        }
                    }
                }
            }
            // The attention over the device's f16 rows, in f64 on the host.
            let ours = download(&out, patches * width);
            let (mut diff, mut norm) = (0f64, 0f64);
            for head in 0..n_head {
                let (q, k, v) = (&host[0][head], &host[1][head], &host[2][head]);
                for i in 0..patches {
                    let scores: Vec<f64> = (0..patches)
                        .map(|j| {
                            (0..d)
                                .map(|t| f64::from(q[i * d + t]) * f64::from(k[j * d + t]))
                                .sum::<f64>()
                                * f64::from(scale)
                        })
                        .collect();
                    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for t in 0..d {
                        let expect =
                            (0..patches).map(|j| weights[j] * f64::from(v[j * d + t])).sum::<f64>()
                                / total;
                        let got = f64::from(ours[i * width + head * d + t]);
                        diff += (got - expect).powi(2);
                        norm += expect * expect;
                    }
                }
            }
            let rel = (diff / norm).sqrt();
            eprintln!(
                "{}: {patches} patches, attention rel {rel:.2e} from the host",
                device.info().name
            );
            assert!(ours.iter().all(|v| v.is_finite()));
            assert!(rel < 2e-3, "{nx} x {ny}: attention rel {rel:.2e}");
        }
    }
}
