//! Tests of the kernels of the mixture of experts (`ExpertKernels`).
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test experts -- --test-threads=1 --nocapture
//! ```
//!
//! `ARES_TEST_GGUF=<first part of Qwen3.8-Flash-Next>` adds the check on the
//! real routers and expert tensors of one layer of each type the file has
//! (gate and up as IQ2_S, IQ2_XXS and IQ1_M; down is Q2_0 everywhere).
//!
//! What is checked, against `ares_quant` as the exact arithmetic:
//!
//! - *The router*: each row's picks and weights equal the reference's
//!   softmax and top K, with equal values in the order of the index.
//! - *The experts compute what they say* (tight): the hidden values the
//!   device quantized equal `silu(gate x^) * (up x^)` of the exact weights and
//!   the dequantized activations `x^` to within their own rounding step, and
//!   the output equals the exact weighted sum of `down h^` over the device's
//!   own quantized hidden values `h^`, to f32 rounding.
//! - *Quantizing costs little* (loose): the output is close to the exact
//!   layer on the unquantized activations. The deviation is printed, so that
//!   a change that makes it worse is noticed.
//! - *A missing expert is waited for on the device* and the result is the
//!   same bits as when it was there from the start.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_cuda::{
    Device, DeviceBuffer, ExpertKernels, ExpertShape, Kernels, PinnedBuffer, RouteRecord, Stream,
    Wait, device_count,
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

    /// A value of a normal distribution, occasionally an outlier: real
    /// activations have a few much larger values per row.
    fn activation(&mut self) -> f32 {
        let u1 = (self.unit() + 1.0) * 0.5 + 1e-7;
        let u2 = self.unit();
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::PI * u2).cos();
        if self.next().is_multiple_of(400) { normal * 30.0 } else { normal }
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// `n_blocks` random blocks of an expert type with finite scales of a size
/// real tensors have.
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for blk in bytes.chunks_exact_mut(block) {
        let scale = 0.002 + 0.02 * (rng.unit() * 0.5 + 0.5);
        if ty == GgmlType::IQ1_M {
            // The f16 d is spread over the top nibbles of the four u16 of the scales.
            let d = ares_quant::f32_to_f16(scale);
            for i in 0..4 {
                let at = 48 + 2 * i;
                let word = u16::from_le_bytes([blk[at], blk[at + 1]]);
                let word = (word & 0x0FFF) | (((d >> (4 * i)) & 0xF) << 12);
                blk[at..at + 2].copy_from_slice(&word.to_le_bytes());
            }
        } else {
            let signed = if rng.next().is_multiple_of(4) { -scale } else { scale };
            blk[0..2].copy_from_slice(&ares_quant::f32_to_f16(signed).to_le_bytes());
        }
    }
    bytes
}

/// A layer to test on: its router and a way to get an expert's matrices.
struct Layer<'a> {
    name: String,
    shape: ExpertShape,
    /// `n_experts` rows of `n_embd`.
    router: Vec<f32>,
    /// Expert -> (gate, up, down) in the file's bytes.
    expert: Box<dyn Fn(u32) -> [Vec<u8>; 3] + 'a>,
}

fn synthetic_layer(gate_ty: GgmlType, seed: u64) -> Layer<'static> {
    let shape = ExpertShape {
        n_embd: 512,
        n_ff: 128,
        n_experts: 96,
        top_k: 4,
        gate_ty,
        down_ty: GgmlType::Q2_0,
    };
    let mut rng = Rng(seed);
    let router = (0..shape.n_experts * shape.n_embd).map(|_| rng.unit() * 0.1).collect();
    let gate_blocks = (shape.n_ff * shape.n_embd / gate_ty.block_values()) as usize;
    let down_blocks = (shape.n_embd * shape.n_ff / 64) as usize;
    let experts: Vec<[Vec<u8>; 3]> = (0..shape.n_experts)
        .map(|_| {
            [
                random_blocks(&mut rng, gate_ty, gate_blocks),
                random_blocks(&mut rng, gate_ty, gate_blocks),
                random_blocks(&mut rng, GgmlType::Q2_0, down_blocks),
            ]
        })
        .collect();
    Layer {
        name: format!("synthetic {gate_ty}"),
        shape,
        router,
        expert: Box::new(move |e| experts[e as usize].clone()),
    }
}

/// The picks of one row by the exact rule: (expert, weight), largest first.
fn reference_route(layer: &Layer<'_>, x: &[f32]) -> (Vec<f32>, Vec<(u32, f32)>) {
    let (n, e, k) =
        (layer.shape.n_embd as usize, layer.shape.n_experts as usize, layer.shape.top_k as usize);
    let logits: Vec<f32> = (0..e)
        .map(|i| {
            layer.router[i * n..(i + 1) * n]
                .iter()
                .zip(x)
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum::<f64>() as f32
        })
        .collect();
    let largest = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits.iter().map(|l| (l - largest).exp()).collect();
    let total: f32 = scaled.iter().sum();
    let prob: Vec<f32> = scaled.iter().map(|s| s / total).collect();
    let mut order: Vec<u32> = (0..e as u32).collect();
    order.sort_by(|a, b| prob[*b as usize].total_cmp(&prob[*a as usize]).then(a.cmp(b)));
    let sum: f32 = order[..k].iter().map(|i| prob[*i as usize]).sum();
    let picks =
        order[..k].iter().map(|i| (*i, prob[*i as usize] / sum.max(1.0 / 16384.0))).collect();
    (prob, picks)
}

/// The activations as the quantizing kernel leaves them: `d * q` per value.
fn dequantized(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (group, out) in x.chunks(32).zip(out.chunks_mut(32)) {
        let amax = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        for (v, o) in group.iter().zip(out) {
            *o = if amax == 0.0 { 0.0 } else { d * (v / d).round() };
        }
    }
    out
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `w` (rows of `n_in` values of `ty`) times `x`, exactly.
fn product(ty: GgmlType, w: &[u8], n_in: usize, x: &[f32]) -> Vec<f64> {
    let row_bytes = ty.bytes_for(n_in as u64).unwrap() as usize;
    let mut row = vec![0.0f32; n_in];
    w.chunks_exact(row_bytes)
        .map(|bytes| {
            ares_quant::dequantize(ty, bytes, &mut row).unwrap();
            row.iter().zip(x).map(|(w, v)| f64::from(*w) * f64::from(*v)).sum()
        })
        .collect()
}

/// What a run of a layer left on the device.
struct Run {
    record: Vec<u8>,
    hidden: Vec<u8>,
    out: Vec<f32>,
    /// Expert -> slot, as the table had it.
    slots: BTreeMap<u32, u32>,
}

/// Device memory of one layer step.
struct Rig {
    stream: Stream,
    kernels: Kernels,
    experts: ExpertKernels,
    router: DeviceBuffer,
    x: DeviceBuffer,
    act: DeviceBuffer,
    logits: DeviceBuffer,
    table: DeviceBuffer,
    record: DeviceBuffer,
    hidden: DeviceBuffer,
    out: DeviceBuffer,
    pool: DeviceBuffer,
}

impl Rig {
    fn new(device: &Device, layer: &Layer<'_>, n_rows: u32, n_slots: u64) -> Self {
        let shape = &layer.shape;
        let router = DeviceBuffer::new(device, shape.router_bytes()).unwrap();
        router.write(0, &f32_bytes(&layer.router)).unwrap();
        let values = u64::from(n_rows) * u64::from(shape.n_embd);
        let table = DeviceBuffer::new(device, 4 * u64::from(shape.n_experts)).unwrap();
        table.write(0, &vec![0xFF; 4 * shape.n_experts as usize]).unwrap();
        Self {
            stream: Stream::new(device, 0).unwrap(),
            kernels: Kernels::load(device).unwrap(),
            experts: ExpertKernels::load(device).unwrap(),
            router,
            x: DeviceBuffer::new(device, values * 4).unwrap(),
            act: DeviceBuffer::new(
                device,
                Kernels::quantized_activation_bytes(shape.n_embd, n_rows),
            )
            .unwrap(),
            logits: DeviceBuffer::new(device, shape.logits_bytes(n_rows)).unwrap(),
            table,
            record: DeviceBuffer::new(device, shape.record_bytes(n_rows)).unwrap(),
            hidden: DeviceBuffer::new(device, shape.hidden_bytes(n_rows)).unwrap(),
            out: DeviceBuffer::new(device, values * 4).unwrap(),
            pool: DeviceBuffer::new(device, n_slots * shape.slot_bytes().unwrap()).unwrap(),
        }
    }

    /// Quantizes the rows and routes them; returns the record.
    fn route(&self, shape: &ExpertShape, x: &[f32], n_rows: u32, token: u32) -> Vec<u8> {
        self.x.write(0, &f32_bytes(x)).unwrap();
        // SAFETY: the buffers were sized for this shape and number of rows.
        unsafe {
            self.kernels
                .quantize_activations(
                    &self.stream,
                    self.x.ptr(),
                    self.act.ptr(),
                    shape.n_embd,
                    n_rows,
                )
                .unwrap();
            self.experts
                .route(
                    &self.stream,
                    shape,
                    self.router.ptr(),
                    self.act.ptr(),
                    self.logits.ptr(),
                    self.table.ptr(),
                    self.record.ptr(),
                    n_rows,
                    token,
                )
                .unwrap();
        }
        self.stream.synchronize().unwrap();
        let mut record = vec![0u8; shape.record_bytes(n_rows) as usize];
        self.record.read(0, &mut record).unwrap();
        record
    }

    fn put_expert(&self, shape: &ExpertShape, slot: u32, matrices: &[Vec<u8>; 3]) {
        let base = u64::from(slot) * shape.slot_bytes().unwrap();
        self.pool.write(base, &matrices[0]).unwrap();
        self.pool.write(base + shape.up_offset().unwrap(), &matrices[1]).unwrap();
        self.pool.write(base + shape.down_offset().unwrap(), &matrices[2]).unwrap();
    }

    fn set_entry(&self, expert: u32, slot: i32) {
        self.table.write(4 * u64::from(expert), &slot.to_le_bytes()).unwrap();
    }

    fn enqueue_experts(&self, shape: &ExpertShape, n_rows: u32, groups: Option<u32>) {
        let zeros = vec![0u8; (n_rows * shape.n_embd * 4) as usize];
        self.out.write(0, &zeros).unwrap();
        // SAFETY: the buffers were sized for this shape and number of rows,
        // and the table names slots of the pool that hold the experts.
        unsafe {
            self.experts
                .experts(
                    &self.stream,
                    shape,
                    self.pool.ptr(),
                    self.table.ptr(),
                    self.record.ptr(),
                    self.act.ptr(),
                    self.hidden.ptr(),
                    self.out.ptr(),
                    n_rows,
                    groups,
                )
                .unwrap();
        }
    }

    fn read_out(&self, shape: &ExpertShape, n_rows: u32) -> (Vec<u8>, Vec<f32>) {
        let mut hidden = vec![0u8; shape.hidden_bytes(n_rows) as usize];
        self.hidden.read(0, &mut hidden).unwrap();
        let mut out = vec![0u8; (n_rows * shape.n_embd * 4) as usize];
        self.out.read(0, &mut out).unwrap();
        (hidden, bytes_f32(&out))
    }
}

/// Routes `x` and computes the experts, with every picked expert in a slot
/// (slots in a scrambled order, so that a slot is never the expert's number).
fn run_layer(device: &Device, layer: &Layer<'_>, x: &[f32], n_rows: u32, groups: bool) -> Run {
    let shape = &layer.shape;
    let rig = Rig::new(device, layer, n_rows, u64::from(shape.top_k * n_rows) + 2);
    let record = rig.route(shape, x, n_rows, 77);
    let parsed = RouteRecord::parse(&record, shape.top_k, n_rows).expect("a record");
    assert_eq!(parsed.token(), 77);
    let picked: Vec<u32> = parsed.groups().map(|g| g.expert).collect();
    assert_eq!(parsed.n_missing(), picked.len(), "the table was empty");
    let n_slots = picked.len() as u32 + 2;
    let mut slots = BTreeMap::new();
    for (i, expert) in picked.iter().enumerate() {
        let slot = (i as u32 * 7 + 3) % n_slots;
        let slot =
            (slot..n_slots).chain(0..slot).find(|s| !slots.values().any(|u| u == s)).unwrap();
        rig.put_expert(shape, slot, &(layer.expert)(*expert));
        rig.set_entry(*expert, slot as i32);
        slots.insert(*expert, slot);
    }
    rig.enqueue_experts(shape, n_rows, groups.then_some(picked.len() as u32));
    rig.stream.synchronize().unwrap();
    let (hidden, out) = rig.read_out(shape, n_rows);
    Run { record, hidden, out, slots }
}

/// The deviations of a run from the references.
struct Deviation {
    /// Largest difference of a hidden value from the reference on the
    /// dequantized activations, in units of the value's own rounding step.
    hidden_steps: f64,
    /// rms of (device - exact sum of down h^ over the device's h^) / rms of that sum.
    tight: f64,
    /// rms of (device - exact layer) / rms of the exact layer.
    loose: f64,
}

fn check_run(layer: &Layer<'_>, x: &[f32], n_rows: u32, run: &Run) -> Deviation {
    let shape = &layer.shape;
    let (n, ff, k) = (shape.n_embd as usize, shape.n_ff as usize, shape.top_k as usize);
    let record = RouteRecord::parse(&run.record, shape.top_k, n_rows).expect("a record");

    // The router: picks and weights of every row.
    for row in 0..n_rows as usize {
        let (prob, reference) = reference_route(layer, &x[row * n..(row + 1) * n]);
        let ids: Vec<u32> = record.ids(row).collect();
        let weights: Vec<f32> = record.weights(row).collect();
        let floor = prob[reference[k - 1].0 as usize];
        let mut seen = Vec::new();
        for (j, (id, weight)) in ids.iter().zip(&weights).enumerate() {
            assert!(!seen.contains(id), "{}: row {row} picked {id} twice", layer.name);
            seen.push(*id);
            let p = prob[*id as usize];
            assert!(
                p >= floor * (1.0 - 1e-4),
                "{}: row {row} picked {id} (prob {p}) below the reference's last ({floor})",
                layer.name
            );
            if j > 0 {
                let before = prob[ids[j - 1] as usize];
                assert!(p <= before * (1.0 + 1e-4), "{}: row {row} picks out of order", layer.name);
            }
            if let Some((_, w)) = reference.iter().find(|(e, _)| e == id) {
                assert!(
                    (weight - w).abs() <= 1e-4 * w.abs(),
                    "{}: row {row} expert {id}: weight {weight}, reference {w}",
                    layer.name
                );
            }
        }
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    // The groups: every picked expert once, ascending, with its rows.
    let groups: Vec<_> = record.groups().collect();
    assert!(groups.windows(2).all(|w| w[0].expert < w[1].expert));
    for row in 0..n_rows as usize {
        for (id, weight) in record.ids(row).zip(record.weights(row)) {
            let group = groups.iter().find(|g| g.expert == id).expect("a group per pick");
            assert!(group.rows >> row & 1 == 1);
            assert_eq!(group.weight[row], weight);
        }
    }
    assert_eq!(groups.iter().map(|g| g.rows.count_ones()).sum::<u32>(), n_rows * shape.top_k);

    // The experts.
    let pairs = k * n_rows as usize;
    let scales_at = (pairs * ff).next_multiple_of(16);
    let mut exact = vec![0.0f64; n_rows as usize * n];
    let mut from_device_hidden = vec![0.0f64; n_rows as usize * n];
    let mut hidden_steps = 0.0f64;
    let mut pair = 0;
    for group in &groups {
        let [gate, up, down] = (layer.expert)(group.expert);
        assert!(run.slots.contains_key(&group.expert));
        for row in (0..n_rows as usize).filter(|b| group.rows >> b & 1 == 1) {
            let xr = &x[row * n..(row + 1) * n];
            let weight = f64::from(group.weight[row]);

            // Exact: unquantized activations, unquantized hidden values.
            let g = product(shape.gate_ty, &gate, n, xr);
            let u = product(shape.gate_ty, &up, n, xr);
            let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| (silu(*g) * u) as f32).collect();
            for (o, d) in
                exact[row * n..(row + 1) * n].iter_mut().zip(product(shape.down_ty, &down, ff, &h))
            {
                *o += weight * d;
            }

            // Tight, stage 1: the device's hidden values against the same
            // formula on the dequantized activations.
            let xq = dequantized(xr);
            let g = product(shape.gate_ty, &gate, n, &xq);
            let u = product(shape.gate_ty, &up, n, &xq);
            let q = &run.hidden[pair * ff..(pair + 1) * ff];
            let scales =
                bytes_f32(&run.hidden[scales_at + pair * (ff / 32) * 8..][..(ff / 32) * 8]);
            let mut device_h = vec![0.0f32; ff];
            for (j, h) in device_h.iter_mut().enumerate() {
                let d = scales[2 * (j / 32)];
                *h = d * f32::from(q[j] as i8);
                let want = silu(g[j]) * u[j];
                // The value was rounded to a multiple of d: half a step, and
                // the f32 rounding of the sums on top.
                let step = f64::from(d).max(1e-12);
                hidden_steps =
                    hidden_steps.max(((f64::from(*h) - want).abs() - 1e-4 * want.abs()) / step);
            }
            for (group_of_32, s) in q.chunks(32).zip(scales.chunks(2)) {
                let codes: i32 = group_of_32.iter().map(|c| i32::from(*c as i8)).sum();
                assert!((s[1] - s[0] * codes as f32).abs() <= 1e-5 * s[1].abs().max(1e-20));
            }

            // Tight, stage 2: down times the device's own hidden values.
            for (o, d) in from_device_hidden[row * n..(row + 1) * n].iter_mut().zip(product(
                shape.down_ty,
                &down,
                ff,
                &device_h,
            )) {
                *o += weight * d;
            }
            pair += 1;
        }
    }
    let rel = |reference: &[f64]| {
        let err: f64 =
            run.out.iter().zip(reference).map(|(o, r)| (f64::from(*o) - r).powi(2)).sum();
        let norm: f64 = reference.iter().map(|r| r * r).sum();
        (err / norm).sqrt()
    };
    Deviation { hidden_steps, tight: rel(&from_device_hidden), loose: rel(&exact) }
}

fn assert_close(layer: &Layer<'_>, device: &Device, n_rows: u32, dev: &Deviation) {
    eprintln!(
        "{} rows {n_rows} on {}: hidden within {:.3} steps, down tight {:.2e}, layer vs exact {:.3} % rms",
        layer.name,
        device.info().name,
        dev.hidden_steps,
        dev.tight,
        dev.loose * 100.0
    );
    assert!(dev.hidden_steps <= 0.51, "hidden values off by {} steps", dev.hidden_steps);
    assert!(dev.tight <= 2e-5, "down product off by {}", dev.tight);
    assert!(dev.loose <= 0.05, "layer off by {} from the exact arithmetic", dev.loose);
}

#[test]
fn router_orders_equal_values_by_index() {
    let Some(gpus) = gpus() else { return };
    // Experts 2 i and 2 i + 1 have the same router row, so the same logit.
    let mut layer = synthetic_layer(GgmlType::IQ2_S, 11);
    let n = layer.shape.n_embd as usize;
    for pair in 0..layer.shape.n_experts as usize / 2 {
        let (a, b) = layer.router.split_at_mut((2 * pair + 1) * n);
        b[..n].copy_from_slice(&a[2 * pair * n..]);
    }
    let mut rng = Rng(5);
    let x: Vec<f32> = (0..3 * n).map(|_| rng.activation()).collect();
    for device in gpus {
        let rig = Rig::new(device, &layer, 3, 1);
        let record = rig.route(&layer.shape, &x, 3, 1);
        let record = RouteRecord::parse(&record, layer.shape.top_k, 3).unwrap();
        for row in 0..3 {
            let ids: Vec<u32> = record.ids(row).collect();
            assert_eq!(ids.len(), 4);
            assert!(ids[0].is_multiple_of(2) && ids[1] == ids[0] + 1, "row {row}: {ids:?}");
            assert!(ids[2].is_multiple_of(2) && ids[3] == ids[2] + 1, "row {row}: {ids:?}");
            let weights: Vec<f32> = record.weights(row).collect();
            assert_eq!(weights[0], weights[1]);
            assert_eq!(weights[2], weights[3]);
        }
    }
}

#[test]
fn experts_match_the_reference_on_generated_blocks() {
    let Some(gpus) = gpus() else { return };
    for (i, gate_ty) in
        [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M].into_iter().enumerate()
    {
        let layer = synthetic_layer(gate_ty, 100 + i as u64);
        let mut rng = Rng(0x9e3779b97f4a7c15 + i as u64);
        for n_rows in [1u32, 2, 5, 8] {
            let n = layer.shape.n_embd as usize;
            let mut x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.activation()).collect();
            if n_rows == 5 {
                // Rows that pick the same experts: one group, several rows.
                let (first, rest) = x.split_at_mut(n);
                rest[..n].copy_from_slice(first);
                rest[2 * n..3 * n].iter_mut().zip(first.iter()).for_each(|(r, f)| *r = *f * 0.5);
            }
            for device in gpus {
                for known_groups in [true, false] {
                    let run = run_layer(device, &layer, &x, n_rows, known_groups);
                    let dev = check_run(&layer, &x, n_rows, &run);
                    assert_close(&layer, device, n_rows, &dev);
                }
            }
        }
    }
}

#[test]
fn a_row_gets_the_same_bits_alone_and_in_a_batch() {
    let Some(gpus) = gpus() else { return };
    let layer = synthetic_layer(GgmlType::IQ2_S, 21);
    let n = layer.shape.n_embd as usize;
    let mut rng = Rng(99);
    let x: Vec<f32> = (0..8 * n).map(|_| rng.activation()).collect();
    for device in gpus {
        let batch = run_layer(device, &layer, &x, 8, false);
        for row in [0usize, 3, 7] {
            let alone = run_layer(device, &layer, &x[row * n..(row + 1) * n], 1, true);
            assert_eq!(alone.out, batch.out[row * n..(row + 1) * n], "row {row}");
        }
    }
}

#[test]
fn a_missing_expert_is_waited_for_on_the_device() {
    let Some(gpus) = gpus() else { return };
    let layer = synthetic_layer(GgmlType::IQ2_XXS, 31);
    let shape = &layer.shape;
    let n = shape.n_embd as usize;
    let mut rng = Rng(7);
    let x: Vec<f32> = (0..2 * n).map(|_| rng.activation()).collect();
    for device in gpus {
        let reference = run_layer(device, &layer, &x, 2, true);

        let rig = Rig::new(device, &layer, 2, 16);
        let record = rig.route(shape, &x, 2, 1);
        let picked: Vec<u32> = RouteRecord::parse(&record, shape.top_k, 2)
            .unwrap()
            .groups()
            .map(|g| g.expert)
            .collect();
        // All but the last picked expert are there.
        let (late, present) = picked.split_last().unwrap();
        for (slot, expert) in present.iter().enumerate() {
            rig.put_expert(shape, slot as u32, &(layer.expert)(*expert));
            rig.set_entry(*expert, slot as i32);
        }
        rig.enqueue_experts(shape, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert!(!rig.stream.is_idle().unwrap(), "the step must wait for the missing expert");

        // The repair, as the store does it: the expert's bytes from pinned
        // memory on another stream, then its table entry.
        let upload = Stream::new(device, 0).unwrap();
        let matrices = (layer.expert)(*late);
        let slot = present.len() as u32;
        let stride = shape.slot_bytes().unwrap();
        let mut staging = PinnedBuffer::new(device, stride as usize + 4).unwrap();
        let offsets = [0, shape.up_offset().unwrap(), shape.down_offset().unwrap()];
        // SAFETY: the pinned buffer holds `stride + 4` bytes and lives until
        // the stream has been synchronized below; the ranges are inside it
        // and inside the pool's slot.
        unsafe {
            let base = staging.as_mut_ptr();
            for (matrix, offset) in matrices.iter().zip(offsets) {
                std::ptr::copy_nonoverlapping(
                    matrix.as_ptr(),
                    base.add(offset as usize),
                    matrix.len(),
                );
                upload
                    .copy_from_host(
                        rig.pool.ptr().offset(u64::from(slot) * stride + offset),
                        base.add(offset as usize),
                        matrix.len(),
                    )
                    .unwrap();
            }
            let entry = base.add(stride as usize);
            std::ptr::copy_nonoverlapping((slot as i32).to_le_bytes().as_ptr(), entry, 4);
            upload.copy_from_host(rig.table.ptr().offset(4 * u64::from(*late)), entry, 4).unwrap();
        }
        rig.stream.synchronize().unwrap();
        upload.synchronize().unwrap();
        let (_, out) = rig.read_out(shape, 2);
        assert_eq!(out, reference.out, "waiting must not change the result");
    }
}

#[test]
fn experts_on_real_tensors() {
    let Some(gpus) = gpus() else { return };
    let Some(path) = std::env::var_os("ARES_TEST_GGUF") else {
        eprintln!("skipped: set ARES_TEST_GGUF=<model.gguf> to check real expert tensors");
        return;
    };
    let gguf = ares_gguf::Gguf::open(&path).unwrap();
    let key = |name: &str| {
        gguf.metadata()
            .find(|(k, _)| k.ends_with(name))
            .and_then(|(_, v)| v.to_string().parse::<u32>().ok())
    };
    let (Some(n_experts), Some(top_k)) = (key(".expert_count"), key(".expert_used_count")) else {
        eprintln!("skipped: the model has no experts");
        return;
    };

    // One layer of each type gate and up have in the file.
    let mut layers: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in gguf.tensors() {
        let Some(rest) = tensor.name.strip_prefix("blk.") else { continue };
        let Some((index, suffix)) = rest.split_once('.') else { continue };
        if suffix == "ffn_gate_exps.weight" {
            layers.entry(tensor.ty.name().to_owned()).or_insert(index.parse().unwrap());
        }
    }
    let mut rng = Rng(0x853c49e6748fea9b);
    for index in layers.into_values() {
        let tensor = |suffix: &str| gguf.tensor(&format!("blk.{index}.{suffix}.weight")).unwrap();
        let (gate, up, down) =
            (tensor("ffn_gate_exps"), tensor("ffn_up_exps"), tensor("ffn_down_exps"));
        let shape = ExpertShape {
            n_embd: gate.dims.get(0) as u32,
            n_ff: gate.dims.get(1) as u32,
            n_experts,
            top_k,
            gate_ty: gate.ty,
            down_ty: down.ty,
        };
        assert_eq!(gate.ty, up.ty);
        if shape.check().is_err() {
            eprintln!("blk.{index}: no expert kernel for {} / {}", gate.ty, down.ty);
            continue;
        }
        let (gate_bytes, down_bytes) =
            (shape.gate_bytes().unwrap() as usize, shape.down_bytes().unwrap() as usize);
        let gguf = &gguf;
        let layer = Layer {
            name: format!("blk.{index} ({} / {})", gate.ty, down.ty),
            shape,
            router: bytes_f32(gguf.data(tensor("ffn_gate_inp"))),
            expert: Box::new(move |e| {
                let e = e as usize;
                [
                    gguf.data(gate)[e * gate_bytes..(e + 1) * gate_bytes].to_vec(),
                    gguf.data(up)[e * gate_bytes..(e + 1) * gate_bytes].to_vec(),
                    gguf.data(down)[e * down_bytes..(e + 1) * down_bytes].to_vec(),
                ]
            }),
        };
        for n_rows in [1u32, 4, 8] {
            let n = shape.n_embd as usize;
            let x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.activation()).collect();
            for device in gpus {
                let run = run_layer(device, &layer, &x, n_rows, n_rows != 4);
                let dev = check_run(&layer, &x, n_rows, &run);
                assert_close(&layer, device, n_rows, &dev);
            }
        }
    }
}
