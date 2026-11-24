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
//! (gate and up as IQ2_S, IQ2_XXS and IQ1_M; down is Q2_0 everywhere). Every other
//! weight type is checked on generated blocks.
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

use ares_core::GgmlType;
use ares_cuda::{
    Device, DeviceBuffer, ExpertKernels, ExpertShape, HostShare, Kernels, NearEntry, PinnedBuffer,
    PredictedList, PrefillPath, Prune, PruneLayer, PruneRule, PruneWeights, RouteRecord, Stream,
};
use common::blocks::{blocks_of, random_blocks_of_one_size, real_blocks};
use common::{Rng, bytes_f32, f32_bytes, gpus};

mod common;

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
    synthetic_layer_of(gate_ty, GgmlType::Q2_0, seed)
}

/// A layer with gate and up of `gate_ty` and down of `down_ty`: rows of 512 values for
/// gate and up, and for down of 128 or of the type's block if that is larger.
fn synthetic_layer_of(gate_ty: GgmlType, down_ty: GgmlType, seed: u64) -> Layer<'static> {
    let shape = ExpertShape {
        n_embd: 512,
        n_ff: down_ty.block_values().max(128),
        n_experts: 96,
        top_k: 4,
        gate_ty,
        down_ty,
    };
    layer_of(
        format!("synthetic {gate_ty} and {down_ty}"),
        shape,
        seed,
        &mut random_blocks_of_one_size,
    )
}

/// A layer of `shape` with a random router and experts whose matrices are the blocks that
/// `blocks(rng, type, number of blocks)` gives.
fn layer_of(
    name: String,
    shape: ExpertShape,
    seed: u64,
    blocks: &mut dyn FnMut(&mut Rng, GgmlType, usize) -> Vec<u8>,
) -> Layer<'static> {
    let (gate_ty, down_ty) = (shape.gate_ty, shape.down_ty);
    let mut rng = Rng(seed);
    let router = (0..shape.n_experts * shape.n_embd).map(|_| rng.unit() * 0.1).collect();
    let gate_blocks = (shape.n_ff * shape.n_embd / gate_ty.block_values()) as usize;
    let down_blocks = (shape.n_embd * shape.n_ff / down_ty.block_values()) as usize;
    let experts: Vec<[Vec<u8>; 3]> = (0..shape.n_experts)
        .map(|_| {
            [
                blocks(&mut rng, gate_ty, gate_blocks),
                blocks(&mut rng, gate_ty, gate_blocks),
                blocks(&mut rng, down_ty, down_blocks),
            ]
        })
        .collect();
    Layer { name, shape, router, expert: Box::new(move |e| experts[e as usize].clone()) }
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
    /// The layer's host share, for the tests in which the host computes,
    /// and what the host sends through: a stream and pinned memory.
    host: Option<DeviceBuffer>,
    reply: Option<(Stream, PinnedBuffer)>,
}

impl Rig {
    /// A rig whose experts may be claimed by the host.
    fn with_host(device: &Device, layer: &Layer<'_>, n_rows: u32, n_slots: u64) -> Self {
        let mut rig = Self::new(device, layer, n_rows, n_slots);
        let bytes = layer.shape.host_share_bytes(n_rows);
        let host = DeviceBuffer::new(device, bytes).unwrap();
        host.write(0, &vec![0u8; bytes as usize]).unwrap();
        rig.host = Some(host);
        rig.reply = Some((
            Stream::new(device, 0).unwrap(),
            PinnedBuffer::new(device, 20 + bytes as usize).unwrap(),
        ));
        rig
    }

    /// The host's side of a step: claims `groups` of the record with
    /// `token`. Like everything the host sends while a step runs, the words
    /// go from pinned memory on a stream of their own: a synchronous copy
    /// would wait for the very kernels that wait for it, and so would
    /// freeing pinned memory, which is why the rig keeps both for its
    /// lifetime. `cell` is the word of the staging memory to send from (0 to
    /// 3), one per call, so that no call rewrites what a copy may still read.
    fn claim(&self, token: u32, groups: &[u32], cell: usize) {
        let host = self.host.as_ref().unwrap();
        let (reply, staging) = self.reply.as_ref().unwrap();
        // SAFETY: the pinned word lives as long as the rig, nothing else
        // writes it, and every claim word is inside the share.
        unsafe {
            let word = staging.as_ptr().add(4 * cell).cast_mut();
            std::ptr::copy_nonoverlapping(HostShare::mark(token).to_le_bytes().as_ptr(), word, 4);
            for group in groups {
                reply
                    .copy_from_host(host.ptr().offset(HostShare::claim_offset(*group)), word, 4)
                    .unwrap();
            }
        }
    }

    /// Delivers `sums` for the record with `token`: the sums, then the mark.
    fn deliver(&self, token: u32, sums: &[f32]) {
        let host = self.host.as_ref().unwrap();
        let (reply, staging) = self.reply.as_ref().unwrap();
        let bytes = f32_bytes(sums);
        // SAFETY: the pinned bytes (behind the four claim words) live as
        // long as the rig, and both ranges are inside the share.
        unsafe {
            let base = staging.as_ptr().add(16).cast_mut();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(4), bytes.len());
            std::ptr::copy_nonoverlapping(HostShare::mark(token).to_le_bytes().as_ptr(), base, 4);
            reply
                .copy_from_host(host.ptr().offset(HostShare::SUM_OFFSET), base.add(4), bytes.len())
                .unwrap();
            reply.copy_from_host(host.ptr().offset(HostShare::DONE_OFFSET), base, 4).unwrap();
        }
    }

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
            host: None,
            reply: None,
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
                    self.host.as_ref().map(DeviceBuffer::ptr),
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
                let steps = ((f64::from(*h) - want).abs() - 1e-4 * want.abs()) / step;
                // A maximum drops what is not a number.
                assert!(steps.is_finite(), "{}: hidden value {j} is {h}, not {want}", layer.name);
                hidden_steps = hidden_steps.max(steps);
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
    assert!(
        dev.hidden_steps.is_finite() && dev.tight.is_finite() && dev.loose.is_finite(),
        "{}: a distance is not a number",
        layer.name
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
fn the_routers_of_later_layers_predict_what_they_would_pick() {
    let Some(gpus) = gpus() else { return };
    let next = synthetic_layer(GgmlType::IQ2_S, 31);
    let after = synthetic_layer(GgmlType::IQ2_S, 32);
    let shape = next.shape;
    let n = shape.n_embd as usize;
    let mut rng = Rng(17);
    for n_rows in [1u32, 3, 8] {
        let x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.activation()).collect();
        for device in gpus {
            let rig = Rig::new(device, &next, n_rows, 1);
            let second = DeviceBuffer::new(device, shape.router_bytes()).unwrap();
            second.write(0, &f32_bytes(&after.router)).unwrap();
            let list_bytes = shape.predict_list_bytes(n_rows);
            let lists = DeviceBuffer::new(device, 2 * list_bytes).unwrap();
            let logits = DeviceBuffer::new(device, 2 * shape.logits_bytes(n_rows)).unwrap();
            // The router of the next layer routes the rows itself as well: its prediction
            // must name what it picks.
            let record = rig.route(&shape, &x, n_rows, 5);
            let record = RouteRecord::parse(&record, shape.top_k, n_rows).unwrap();
            // SAFETY: the buffers were sized for this shape and number of rows.
            unsafe {
                rig.experts
                    .predict(
                        &rig.stream,
                        &shape,
                        &[rig.router.ptr(), second.ptr()],
                        rig.act.ptr(),
                        logits.ptr(),
                        lists.ptr(),
                        list_bytes,
                        n_rows,
                        99,
                    )
                    .unwrap();
            }
            rig.stream.synchronize().unwrap();
            let mut host = vec![0u8; 2 * list_bytes as usize];
            lists.read(0, &mut host).unwrap();
            for (d, layer) in [&next, &after].into_iter().enumerate() {
                let bytes = &host[d * list_bytes as usize..(d + 1) * list_bytes as usize];
                assert!(PredictedList::parse(bytes, 98).is_none(), "a list of another step");
                let list = PredictedList::parse(bytes, 99).expect("a list of this step");
                // The reference: every expert some row picks, by its best rank, then number.
                let mut best = BTreeMap::new();
                for row in 0..n_rows as usize {
                    let (_, picks) = reference_route(layer, &x[row * n..(row + 1) * n]);
                    for (rank, (expert, _)) in picks.iter().enumerate() {
                        let at = best.entry(*expert).or_insert(rank);
                        *at = (*at).min(rank);
                    }
                }
                let mut want: Vec<(usize, u32)> = best.into_iter().map(|(e, r)| (r, e)).collect();
                want.sort_unstable();
                let want: Vec<u32> = want.into_iter().map(|(_, e)| e).collect();
                let got: Vec<u32> = list.experts().collect();
                assert_eq!(got, want, "router {d}, {n_rows} rows on {}", device.info().name);
                if d == 0 {
                    let mut picked: Vec<u32> =
                        (0..n_rows as usize).flat_map(|row| record.ids(row)).collect();
                    picked.sort_unstable();
                    picked.dedup();
                    let mut named = got.clone();
                    named.sort_unstable();
                    assert_eq!(named, picked, "what the router picks when its layer runs");
                }
            }
            // One router alone.
            // SAFETY: as above.
            unsafe {
                rig.experts
                    .predict(
                        &rig.stream,
                        &shape,
                        &[second.ptr()],
                        rig.act.ptr(),
                        logits.ptr(),
                        lists.ptr(),
                        list_bytes,
                        n_rows,
                        100,
                    )
                    .unwrap();
            }
            rig.stream.synchronize().unwrap();
            lists.read(0, &mut host).unwrap();
            let one = PredictedList::parse(&host[..list_bytes as usize], 100).unwrap();
            let two = PredictedList::parse(&host[list_bytes as usize..], 99).unwrap();
            assert!(one.experts().eq(two.experts()), "the second list of before is untouched");
        }
    }
}

#[test]
fn the_router_routes_more_rows_than_a_decode_step_takes() {
    let Some(gpus) = gpus() else { return };
    let layer = synthetic_layer(GgmlType::IQ2_S, 21);
    let shape = &layer.shape;
    let (n, k) = (shape.n_embd as usize, shape.top_k as usize);
    let mut rng = Rng(9);
    for n_rows in [9u32, 17, 70] {
        let x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.activation()).collect();
        for device in gpus {
            let rig = Rig::new(device, &layer, n_rows, 1);
            rig.x.write(0, &f32_bytes(&x)).unwrap();
            // SAFETY: the buffers were sized for this shape and number of rows.
            unsafe {
                let quantize = rig.kernels.quantize_activations(
                    &rig.stream,
                    rig.x.ptr(),
                    rig.act.ptr(),
                    shape.n_embd,
                    n_rows,
                );
                assert!(quantize.is_err(), "the decode quantization takes eight rows");
                rig.kernels
                    .prepare_prefill_activations(
                        &rig.stream,
                        PrefillPath::Int8,
                        rig.x.ptr(),
                        rig.act.ptr(),
                        shape.n_embd,
                        n_rows,
                    )
                    .unwrap();
                rig.experts
                    .logits_rows(
                        &rig.stream,
                        shape,
                        rig.router.ptr(),
                        rig.act.ptr(),
                        rig.logits.ptr(),
                        n_rows,
                    )
                    .unwrap();
                rig.experts
                    .pick_rows(&rig.stream, shape, rig.logits.ptr(), rig.record.ptr(), n_rows)
                    .unwrap();
            }
            rig.stream.synchronize().unwrap();
            let mut record = vec![0u8; shape.record_bytes(n_rows) as usize];
            rig.record.read(0, &mut record).unwrap();
            let cap = n_rows as usize * k;
            let ids = &record[16..16 + 4 * cap];
            let weights = bytes_f32(&record[16 + 4 * cap..16 + 8 * cap]);
            for row in 0..n_rows as usize {
                let (_, expected) = reference_route(&layer, &x[row * n..(row + 1) * n]);
                for (j, (expert, weight)) in expected.iter().enumerate() {
                    let at = row * k + j;
                    let id = u32::from_le_bytes(ids[4 * at..4 * at + 4].try_into().unwrap());
                    assert_eq!(id, *expert, "row {row} of {n_rows}, pick {j}");
                    assert!(
                        (weights[at] - weight).abs() <= 1e-4 * weight.max(1e-3),
                        "row {row} of {n_rows}, pick {j}: weight {} against {weight}",
                        weights[at]
                    );
                }
            }
        }
    }
}

/// What a pruned row computes by the rules of `kernels/expert_router.cuh`, from the
/// router's picks and weights (`ids`, `weights`, as the record has them), every expert's
/// probability, which experts are in a tier, and each expert's list of most similar ones:
/// (expert, weight) in rank order.
fn reference_prune(
    prune: &Prune,
    ids: &[u32],
    weights: &[f32],
    prob: &[f32],
    cached: &[bool],
    near: &[Vec<NearEntry>],
) -> Vec<(u32, f32)> {
    let k = ids.len();
    let mut touched: Vec<bool> = (0..k)
        .map(|j| {
            !cached[ids[j] as usize]
                && j as u32 >= prune.from_rank
                && weights[j] < prune.below_weight
        })
        .collect();
    let mut stand: Vec<Option<(u32, f32)>> = vec![None; k];
    match prune.rule {
        PruneRule::Skip => {}
        PruneRule::Router => {
            let mut candidates: Vec<u32> = (0..prob.len() as u32)
                .filter(|e| cached[*e as usize] && !ids.contains(e))
                .collect();
            candidates
                .sort_by(|a, b| prob[*b as usize].total_cmp(&prob[*a as usize]).then(a.cmp(b)));
            let mut next = candidates.into_iter();
            for j in 0..k {
                if touched[j] {
                    stand[j] = next.next().map(|e| (e, 1.0));
                }
            }
        }
        PruneRule::Similar => {
            for j in 0..k {
                if !touched[j] {
                    continue;
                }
                let found = near[ids[j] as usize]
                    .iter()
                    .take_while(|n| {
                        (n.closeness.clamp(0.0, 1.0) * 65535.0).round()
                            >= (prune.min_closeness.clamp(0.0, 1.0) * 65535.0).round()
                    })
                    .find(|n| {
                        let y = u32::from(n.expert);
                        cached[y as usize]
                            && !ids.contains(&y)
                            && !stand.iter().flatten().any(|(s, _)| *s == y)
                    });
                stand[j] = found.map(|n| (u32::from(n.expert), n.scale));
                touched[j] = found.is_some();
            }
        }
    }
    let mut computed: Vec<(u32, f32, f32)> = Vec::new();
    for j in 0..k {
        if !touched[j] {
            computed.push((ids[j], prob[ids[j] as usize], weights[j]));
        } else if let Some((y, scale)) = stand[j] {
            let kept =
                if prune.weights == PruneWeights::Scaled { weights[j] * scale } else { weights[j] };
            computed.push((y, prob[y as usize], kept));
        }
    }
    let sum: f32 = computed.iter().map(|c| c.1).sum();
    computed
        .into_iter()
        .map(|(e, p, kept)| {
            (
                e,
                if prune.weights == PruneWeights::Renormalise {
                    p / sum.max(1.0 / 16384.0)
                } else {
                    kept
                },
            )
        })
        .collect()
}

#[test]
fn a_pruned_router_computes_what_its_rule_says() {
    let Some(gpus) = gpus() else { return };
    let layer = synthetic_layer(GgmlType::IQ2_S, 31);
    let shape = &layer.shape;
    let (n, k, e) = (shape.n_embd as usize, shape.top_k as usize, shape.n_experts as usize);
    let mut rng = Rng(77);
    // Every other expert or so is in a tier; each expert's list of similar
    // ones is six others with falling closeness.
    let cached: Vec<bool> = (0..e).map(|_| rng.next() % 5 < 2).collect();
    let near_count = 6usize;
    let near: Vec<Vec<NearEntry>> = (0..e)
        .map(|x| {
            let mut closeness = 0.95;
            (0..near_count)
                .map(|i| {
                    closeness -= 0.1 + 0.05 * (rng.unit() + 1.0);
                    let y = (x + 1 + (rng.next() as usize + i * 7) % (e - 1)) % e;
                    NearEntry { expert: y as u16, closeness, scale: 0.5 + (rng.unit() + 1.0) }
                })
                .collect()
        })
        .collect();
    let cached_bytes: Vec<u8> = cached.iter().map(|c| u8::from(*c)).collect();
    let near_bytes: Vec<u8> = near.iter().flatten().flat_map(|n| n.to_bytes()).collect();
    let mut dials = Vec::new();
    for rule in [PruneRule::Skip, PruneRule::Router, PruneRule::Similar] {
        for weights in [PruneWeights::Keep, PruneWeights::Renormalise, PruneWeights::Scaled] {
            dials.push(Prune { weights, ..Prune::new(rule, 1) });
        }
    }
    dials.push(Prune { below_weight: 0.3, ..Prune::new(PruneRule::Skip, 0) });
    dials.push(Prune { min_closeness: 0.6, ..Prune::new(PruneRule::Similar, 0) });
    // Every expert in a tier: nothing is touched.
    let all = vec![1u8; e];
    for n_rows in [1u32, 3, 8] {
        let x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.activation()).collect();
        for device in gpus {
            let rig = Rig::new(device, &layer, n_rows, 1);
            let plain = rig.route(shape, &x, n_rows, 5);
            let tables = DeviceBuffer::new(device, (2 * e + near_bytes.len()) as u64).unwrap();
            tables.write(0, &cached_bytes).unwrap();
            tables.write(e as u64, &all).unwrap();
            tables.write(2 * e as u64, &near_bytes).unwrap();
            let route = |prune: &Prune, cached_at: u64| {
                let layer = PruneLayer {
                    cached: tables.ptr().offset(cached_at),
                    near: tables.ptr().offset(2 * e as u64),
                    near_count: near_count as u32,
                };
                // SAFETY: the buffers were sized for this shape and number of rows.
                unsafe {
                    rig.kernels
                        .quantize_activations(
                            &rig.stream,
                            rig.x.ptr(),
                            rig.act.ptr(),
                            shape.n_embd,
                            n_rows,
                        )
                        .unwrap();
                    rig.experts
                        .route_pruned(
                            &rig.stream,
                            shape,
                            rig.router.ptr(),
                            rig.act.ptr(),
                            rig.logits.ptr(),
                            rig.table.ptr(),
                            rig.record.ptr(),
                            n_rows,
                            5,
                            prune,
                            &layer,
                        )
                        .unwrap();
                }
                rig.stream.synchronize().unwrap();
                let mut record = vec![0u8; shape.record_bytes(n_rows) as usize];
                rig.record.read(0, &mut record).unwrap();
                record
            };
            // Nothing in the file only: the record of the plain router, bit for bit.
            for prune in &dials {
                assert_eq!(
                    route(prune, e as u64),
                    plain,
                    "{prune:?}, {n_rows} rows, all in a tier"
                );
            }
            let mut touched_rows = 0;
            for prune in &dials {
                let record = route(prune, 0);
                let parsed = RouteRecord::parse(&record, shape.top_k, n_rows).unwrap();
                let plain = RouteRecord::parse(&plain, shape.top_k, n_rows).unwrap();
                let groups: Vec<_> = parsed.groups().collect();
                assert!(groups.windows(2).all(|w| w[0].expert < w[1].expert));
                for row in 0..n_rows as usize {
                    // The record keeps the router's picks.
                    let ids: Vec<u32> = parsed.ids(row).collect();
                    let weights: Vec<f32> = parsed.weights(row).collect();
                    assert_eq!(ids, plain.ids(row).collect::<Vec<_>>());
                    assert_eq!(weights, plain.weights(row).collect::<Vec<_>>());
                    let (prob, _) = reference_route(&layer, &x[row * n..(row + 1) * n]);
                    let mut want = reference_prune(prune, &ids, &weights, &prob, &cached, &near);
                    want.sort_by_key(|w| w.0);
                    let got: Vec<(u32, f32)> = groups
                        .iter()
                        .filter(|g| g.rows >> row & 1 == 1)
                        .map(|g| (g.expert, g.weight[row]))
                        .collect();
                    if want.len() != k || want.iter().any(|w| !ids.contains(&w.0)) {
                        touched_rows += 1;
                    }
                    assert_eq!(
                        got.iter().map(|g| g.0).collect::<Vec<_>>(),
                        want.iter().map(|w| w.0).collect::<Vec<_>>(),
                        "{prune:?}, row {row} of {n_rows}: picks {ids:?}"
                    );
                    for ((_, got), (expert, want)) in got.iter().zip(&want) {
                        assert!(
                            (got - want).abs() <= 1e-4 * want.abs().max(1e-3),
                            "{prune:?}, row {row} of {n_rows}, expert {expert}: {got} against {want}"
                        );
                    }
                }
                let pairs: u32 = groups.iter().map(|g| g.rows.count_ones()).sum();
                let missing = groups.iter().filter(|g| g.slot.is_none()).count();
                assert_eq!(parsed.n_missing(), missing);
                assert!(pairs <= n_rows * shape.top_k);
            }
            assert!(touched_rows > 0, "the rules touched nothing in {n_rows} rows");
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

/// Every weight type as the type of gate and up (with down as Q2_0) and as the type of
/// down (with gate and up as IQ2_S), and a few pairs of other types: the same checks as on
/// the types of the model the kernels were written for.
#[test]
fn experts_of_every_weight_type_match_the_reference() {
    let Some(gpus) = gpus() else { return };
    let mut pairs: Vec<(GgmlType, GgmlType)> = Vec::new();
    for ty in ExpertShape::types() {
        pairs.push((ty, GgmlType::Q2_0));
        pairs.push((GgmlType::IQ2_S, ty));
    }
    pairs.extend([
        (GgmlType::Q4_K, GgmlType::Q4_K),
        (GgmlType::F16, GgmlType::BF16),
        (GgmlType::Q8_0, GgmlType::Q6_K),
        (GgmlType::TQ1_0, GgmlType::IQ1_S),
    ]);
    for (i, (gate_ty, down_ty)) in pairs.into_iter().enumerate() {
        let layer = synthetic_layer_of(gate_ty, down_ty, 300 + i as u64);
        check_layer_of_types(gpus, &layer, 0x9e3779b97f4a7c15 + i as u64, &[1, 5, 8]);
    }
}

/// Runs `layer` with steps of `rows` rows on every device and holds the result to the
/// bounds of the types the kernels were written for.
fn check_layer_of_types(gpus: &[Device], layer: &Layer<'_>, seed: u64, rows: &[u32]) {
    layer.shape.check().expect("a shape the kernels take");
    let mut rng = Rng(seed);
    let n = layer.shape.n_embd as usize;
    for &n_rows in rows {
        let mut x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.activation()).collect();
        if n_rows == 5 {
            // Rows that pick the same experts: one group, several rows.
            let (first, rest) = x.split_at_mut(n);
            rest[..n].copy_from_slice(first);
        }
        for device in gpus {
            let run = run_layer(device, layer, &x, n_rows, false);
            assert!(
                run.out.iter().all(|v| v.is_finite()),
                "{}: an output is not a number",
                layer.name
            );
            let dev = check_run(layer, &x, n_rows, &run);
            assert_close(layer, device, n_rows, &dev);
        }
    }
}

/// The widths of an expert of the model the kernels were written for: rows of 2560 values
/// for gate and up, and of 640 for down, or 768 for a type of down whose blocks of 256
/// values do not divide 640.
fn wide_shape(gate_ty: GgmlType, down_ty: GgmlType) -> ExpertShape {
    let n_ff = if 640 % down_ty.block_values() == 0 { 640 } else { 768 };
    ExpertShape { n_embd: 2560, n_ff, n_experts: 12, top_k: 2, gate_ty, down_ty }
}

/// Every weight type at the widths of a real expert, as the type of gate and up and as the
/// type of down: rows of many blocks, and row lengths in bytes that are not multiples of
/// 16 for most types.
#[test]
fn experts_of_every_weight_type_match_the_reference_at_real_widths() {
    let Some(gpus) = gpus() else { return };
    for (i, ty) in ExpertShape::types().enumerate() {
        for (gate_ty, down_ty) in [(ty, GgmlType::Q2_0), (GgmlType::IQ2_S, ty)] {
            let shape = wide_shape(gate_ty, down_ty);
            let name = format!("{gate_ty} and {down_ty}, {} x {}", shape.n_embd, shape.n_ff);
            let layer = layer_of(name, shape, 700 + i as u64, &mut random_blocks_of_one_size);
            check_layer_of_types(gpus, &layer, 0x2545f4914f6cdd1d + i as u64, &[1, 8]);
        }
    }
}

/// `ARES_TEST_GGUF=<any model file>`: the blocks of the file's tensors as experts. Every
/// weight type the file has is once the type of gate and up and once the type of down, at
/// the widths of a real expert, with the blocks of its largest tensor of that type as the
/// matrices (what a quantizer writes: the scales, the codes and their statistics, which
/// generated blocks do not have).
#[test]
fn experts_of_the_blocks_of_real_tensors_match_the_reference() {
    let Some(gpus) = gpus() else { return };
    let Some(path) = std::env::var_os("ARES_TEST_GGUF") else {
        eprintln!("skipped: set ARES_TEST_GGUF=<model.gguf> to check blocks of real tensors");
        return;
    };
    let gguf = ares_gguf::Gguf::open(&path).unwrap();
    for (i, (ty, data)) in real_blocks(&gguf).into_iter().enumerate() {
        for (gate_ty, down_ty) in [(ty, GgmlType::Q2_0), (GgmlType::IQ2_S, ty)] {
            let shape = wide_shape(gate_ty, down_ty);
            let name = format!("real blocks of {ty} as {gate_ty} and {down_ty}");
            let layer = layer_of(name, shape, 900 + i as u64, &mut blocks_of(ty, data));
            check_layer_of_types(gpus, &layer, 0x853c49e6748fea9b + i as u64, &[1, 8]);
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

/// The host computes an expert instead of uploading it: the kernels leave
/// the claimed group out, wait for the host's sums and add them.
#[test]
fn a_claimed_expert_comes_from_the_host() {
    let Some(gpus) = gpus() else { return };
    let layer = synthetic_layer(GgmlType::IQ2_S, 47);
    let shape = &layer.shape;
    let n = shape.n_embd as usize;
    let mut rng = Rng(11);
    let x: Vec<f32> = (0..2 * n).map(|_| rng.activation()).collect();
    for device in gpus {
        let reference = run_layer(device, &layer, &x, 2, true);

        // One step with the experts in `resident` in slots and the others
        // claimed, the host delivering `sums`. Returns the output and
        // whether the step waited for the host.
        let run = |token: u32, resident: &dyn Fn(usize, usize) -> bool, sums: &[f32]| {
            let rig = Rig::with_host(device, &layer, 2, 16);
            let record = rig.route(shape, &x, 2, token);
            let picked: Vec<u32> = RouteRecord::parse(&record, shape.top_k, 2)
                .unwrap()
                .groups()
                .map(|g| g.expert)
                .collect();
            let mut claimed = Vec::new();
            for (group, expert) in picked.iter().enumerate() {
                if resident(group, picked.len()) {
                    rig.put_expert(shape, group as u32, &(layer.expert)(*expert));
                    rig.set_entry(*expert, group as i32);
                } else {
                    claimed.push(group as u32);
                }
            }
            rig.enqueue_experts(shape, 2, None);
            std::thread::sleep(std::time::Duration::from_millis(30));
            let waited_for_claim = !rig.stream.is_idle().unwrap();
            // A claim of an earlier record must not count.
            rig.claim(token.wrapping_sub(1), &claimed, 0);
            std::thread::sleep(std::time::Duration::from_millis(10));
            assert_eq!(waited_for_claim, !rig.stream.is_idle().unwrap());
            rig.claim(token, &claimed, 1);
            std::thread::sleep(std::time::Duration::from_millis(30));
            let waited_for_sums = !rig.stream.is_idle().unwrap();
            rig.deliver(token, sums);
            rig.stream.synchronize().unwrap();
            let (_, out) = rig.read_out(shape, 2);
            (out, waited_for_claim, waited_for_sums)
        };

        let zeros = vec![0.0f32; 2 * n];
        let sums: Vec<f32> = (0..2 * n).map(|_| rng.activation()).collect();
        let last = |group: usize, groups: usize| group + 1 < groups;
        // Everything in slots: the share is never read, nothing waits.
        let (all, waited, waited_for_sums) = run(5, &|_, _| true, &sums);
        assert!(!waited && !waited_for_sums, "a layer without a miss must not wait for the host");
        assert_eq!(all, reference.out, "a host share must not change a layer without a miss");
        // The last group claimed: the step waits for the claim, then for
        // the sums, and adds exactly them.
        let (without, waited, waited_for_sums) = run(6, &last, &zeros);
        assert!(waited && waited_for_sums, "the step must wait for the claim and for the sums");
        let (with, _, _) = run(7, &last, &sums);
        let expected: Vec<f32> = without.iter().zip(&sums).map(|(o, s)| o + s).collect();
        assert_eq!(with, expected, "the host's sums are added to the rows as they are");
        // The claimed group alone, by the device: both halves together are
        // the whole layer, to the rounding of a different order of additions.
        let (only, _, _) = run(8, &|group, groups| group + 1 == groups, &zeros);
        let norm = reference.out.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        let err = (reference.out.iter().zip(without.iter().zip(&only)))
            .map(|(r, (a, b))| (f64::from(*r) - f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>();
        assert!((err / norm).sqrt() < 1e-6, "the claimed and the computed part do not add up");
        assert!(only.iter().any(|v| *v != 0.0), "the claimed expert contributes something");
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
