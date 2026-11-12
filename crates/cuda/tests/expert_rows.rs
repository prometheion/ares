//! Tests of the many-row kernels of the mixture of experts (`ExpertRowKernels`).
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test --release -p ares-cuda --test expert_rows -- --test-threads=1 --nocapture
//! ```
//!
//! `ARES_TEST_GGUF=<first part of Qwen3.8-Flash-Next>` adds the check on the real expert
//! tensors of one layer of each type the file has.
//!
//! What is checked, against `ares_quant` and f64 arithmetic as the exact result:
//!
//! - *The plan* equals the sort a CPU makes: counts, the first position of each expert,
//!   the items, the rows and weights at each position, the position of each pair; for
//!   plans of one block of pairs and of several, one expert taking every row, every
//!   row a different expert, counts that are not multiples of the tile, ids out of
//!   range.
//! - *The products*: the hidden values and the output of every type and tile against
//!   the exact layer, and the output of the ordered sum bit for bit alone and in a batch.
//! - *Missing experts* (no address) add nothing, are reported and are not read.

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_cuda::{
    Accumulate, Device, DeviceBuffer, ExpertRowKernels, ExpertShape, Kernels, PrefillPath,
    ROWS_TILES, RowsPlan, Stream, Wait, device_count,
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
            let all: Vec<Device> =
                (0..count).map(|i| Device::open(i, Wait::Block).expect("open device")).collect();
            // The many-row kernels need compute capability 8.0.
            Some(all.into_iter().filter(|d| d.info().compute_capability.0 >= 8).collect())
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

    /// A value of a normal distribution, occasionally an outlier: real activations have
    /// a few much larger values per row.
    fn activation(&mut self) -> f32 {
        let u1 = (self.unit() + 1.0) * 0.5 + 1e-7;
        let u2 = self.unit();
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::PI * u2).cos();
        if self.next().is_multiple_of(400) { normal * 30.0 } else { normal }
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() >> 33) as u32 % n
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// The weight types, each with the byte offsets of the f16 scales of a block; IQ1_M,
/// MXFP4 and NVFP4 keep theirs in other forms and the float types have none
/// (`random_blocks`).
const TYPES: [(GgmlType, &[usize]); 28] = [
    (GgmlType::Q4_K, &[0, 2]),
    (GgmlType::Q5_K, &[0, 2]),
    (GgmlType::Q6_K, &[208]),
    (GgmlType::Q3_K, &[108]),
    (GgmlType::Q2_K, &[80, 82]),
    (GgmlType::IQ4_XS, &[0]),
    (GgmlType::IQ4_NL, &[0]),
    (GgmlType::IQ3_S, &[0]),
    (GgmlType::IQ3_XXS, &[0]),
    (GgmlType::IQ2_S, &[0]),
    (GgmlType::IQ2_XS, &[0]),
    (GgmlType::IQ2_XXS, &[0]),
    (GgmlType::IQ1_S, &[0]),
    (GgmlType::IQ1_M, &[]),
    (GgmlType::Q8_0, &[0]),
    (GgmlType::Q4_0, &[0]),
    (GgmlType::Q4_1, &[0, 2]),
    (GgmlType::Q5_0, &[0]),
    (GgmlType::Q5_1, &[0, 2]),
    (GgmlType::TQ1_0, &[52]),
    (GgmlType::TQ2_0, &[64]),
    (GgmlType::MXFP4, &[]),
    (GgmlType::NVFP4, &[]),
    (GgmlType::Q1_0, &[0]),
    (GgmlType::Q2_0, &[0]),
    (GgmlType::F32, &[]),
    (GgmlType::F16, &[]),
    (GgmlType::BF16, &[]),
];

/// The scale of a made-up block: between 0.002 and 0.022 for the types whose codes are
/// small numbers, as the blocks of real tensors have, and less by the size of the codes
/// for the types whose codes and inner scales are large (a block of Q6_K decodes to up to
/// 4096 times its scale), so that the weights of every type are of one size.
fn block_scale(rng: &mut Rng, ty: GgmlType) -> f32 {
    let codes = match ty {
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ3_S => 8.0,
        GgmlType::Q6_K | GgmlType::IQ4_XS => 24.0,
        GgmlType::IQ3_XXS => 3.0,
        _ => 1.0,
    };
    (0.002 + 0.02 * (rng.unit() * 0.5 + 0.5)) / codes
}

/// `n_blocks` random blocks of a weight type with finite scales (any other bytes are valid
/// in every one of these types); for a float type, `n_blocks` values.
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    let (_, scales) = TYPES.iter().find(|(t, _)| *t == ty).expect("a weight type");
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for blk in bytes.chunks_exact_mut(block) {
        let scale = block_scale(rng, ty);
        for &at in *scales {
            let signed = if rng.next().is_multiple_of(4) { -scale } else { scale };
            blk[at..at + 2].copy_from_slice(&ares_quant::f32_to_f16(signed).to_le_bytes());
        }
        match ty {
            // The f16 d is spread over the top nibbles of the four u16 of the scales.
            GgmlType::IQ1_M => {
                let d = ares_quant::f32_to_f16(scale);
                for i in 0..4 {
                    let at = 48 + 2 * i;
                    let word = u16::from_le_bytes([blk[at], blk[at + 1]]);
                    let word = (word & 0x0FFF) | (((d >> (4 * i)) & 0xF) << 12);
                    blk[at..at + 2].copy_from_slice(&word.to_le_bytes());
                }
            }
            // An E8M0 exponent: 2^-7 to 2^-4.
            GgmlType::MXFP4 => blk[0] = 121 + (rng.next() % 4) as u8,
            // Four UE4M3 scales of exponent 1 to 4 (2^-7 to 2^-4), never the NaN code.
            GgmlType::NVFP4 => {
                for byte in blk.iter_mut().take(4) {
                    *byte = ((1 + rng.next() % 4) << 3 | (rng.next() % 8)) as u8;
                }
            }
            GgmlType::F32 => blk.copy_from_slice(&(rng.unit() * 0.2).to_le_bytes()),
            GgmlType::F16 => {
                blk.copy_from_slice(&ares_quant::f32_to_f16(rng.unit() * 0.2).to_le_bytes());
            }
            GgmlType::BF16 => {
                blk.copy_from_slice(&ares_quant::f32_to_bf16(rng.unit() * 0.2).to_le_bytes());
            }
            _ => {}
        }
    }
    bytes
}

/// A layer to test on: its shape and a way to get an expert's matrices.
struct Layer<'a> {
    name: String,
    shape: ExpertShape,
    /// Expert -> (gate, up, down) in the file's bytes.
    expert: Box<dyn Fn(u32) -> [Vec<u8>; 3] + 'a>,
}

fn synthetic_layer(gate_ty: GgmlType, n_experts: u32, top_k: u32, seed: u64) -> Layer<'static> {
    synthetic_layer_of(gate_ty, GgmlType::Q2_0, n_experts, top_k, seed)
}

/// A layer with gate and up of `gate_ty` and down of `down_ty`: rows of 512 values for
/// gate and up, and for down of 128 or of the type's block if that is larger.
fn synthetic_layer_of(
    gate_ty: GgmlType,
    down_ty: GgmlType,
    n_experts: u32,
    top_k: u32,
    seed: u64,
) -> Layer<'static> {
    let n_ff = down_ty.block_values().max(128);
    let shape = ExpertShape { n_embd: 512, n_ff, n_experts, top_k, gate_ty, down_ty };
    layer_of(format!("synthetic {gate_ty} and {down_ty}"), shape, seed, &mut random_blocks)
}

/// A layer of `shape` with experts whose matrices are the blocks that
/// `blocks(rng, type, number of blocks)` gives.
fn layer_of(
    name: String,
    shape: ExpertShape,
    seed: u64,
    blocks: &mut dyn FnMut(&mut Rng, GgmlType, usize) -> Vec<u8>,
) -> Layer<'static> {
    let (gate_ty, down_ty) = (shape.gate_ty, shape.down_ty);
    let mut rng = Rng(seed);
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
    Layer { name, shape, expert: Box::new(move |e| experts[e as usize].clone()) }
}

/// The picks of the rows: `n_rows * top_k` expert ids and as many weights.
struct Picks {
    ids: Vec<u32>,
    weights: Vec<f32>,
}

impl Picks {
    /// The route record the plan reads: a header of 16 bytes, the ids, the weights.
    fn record(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; 16];
        bytes.extend(self.ids.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend(f32_bytes(&self.weights));
        bytes
    }
}

enum Pattern {
    /// Each row picks `top_k` different experts at random.
    Random,
    /// Every row picks expert 5 first, and others at random.
    OneExpertTakesEveryRow,
    /// Row `r` picks experts `r * top_k ..`: no expert has two pairs.
    EveryRowDifferent,
    /// The first rows pick all the same experts, the rest random: runs that are not
    /// whole tiles.
    Skewed,
}

fn picks(pattern: &Pattern, shape: &ExpertShape, n_rows: u32, rng: &mut Rng) -> Picks {
    let k = shape.top_k;
    let mut ids = Vec::new();
    let mut weights = Vec::new();
    for row in 0..n_rows {
        let mut chosen: Vec<u32> = Vec::new();
        match pattern {
            Pattern::Random => {}
            Pattern::OneExpertTakesEveryRow => chosen.push(5),
            Pattern::EveryRowDifferent => {
                chosen.extend((0..k).map(|j| (row * k + j) % shape.n_experts));
            }
            Pattern::Skewed => {
                if row % 3 != 2 {
                    chosen.extend([2, 3]);
                }
            }
        }
        while chosen.len() < k as usize {
            let e = rng.below(shape.n_experts);
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        let raw: Vec<f32> = (0..k).map(|_| 0.05 + (rng.unit() + 1.0)).collect();
        let sum: f32 = raw.iter().sum();
        ids.extend(chosen);
        weights.extend(raw.iter().map(|w| w / sum));
    }
    Picks { ids, weights }
}

/// The plan a CPU makes: for each expert its pairs in order, as (row, weight, pair).
fn cpu_plan(shape: &ExpertShape, picks: &Picks) -> Vec<Vec<(u32, f32, u32)>> {
    let mut by_expert = vec![Vec::new(); shape.n_experts as usize];
    for (pair, (id, weight)) in picks.ids.iter().zip(&picks.weights).enumerate() {
        if *id < shape.n_experts {
            by_expert[*id as usize].push((pair as u32 / shape.top_k, *weight, pair as u32));
        }
    }
    by_expert
}

fn check_plan(shape: &ExpertShape, picks: &Picks, plan: &RowsPlan<'_>, tile: u32, what: &str) {
    let by_expert = cpu_plan(shape, picks);
    let n_valid: usize = by_expert.iter().map(Vec::len).sum();
    assert_eq!(plan.n_valid() as usize, n_valid, "{what}: valid pairs");
    let mut first = 0u32;
    let mut items = plan.items();
    for (expert, pairs) in by_expert.iter().enumerate() {
        assert_eq!(plan.first(expert as u32), first, "{what}: first position of expert {expert}");
        assert_eq!(plan.count(expert as u32) as usize, pairs.len(), "{what}: count of {expert}");
        for (i, (row, weight, pair)) in pairs.iter().enumerate() {
            let pos = first + i as u32;
            assert_eq!(plan.pair_row(pos), *row, "{what}: row at {pos}");
            assert_eq!(plan.pair_weight(pos), *weight, "{what}: weight at {pos}");
            assert_eq!(plan.position(*pair), Some(pos), "{what}: position of pair {pair}");
        }
        let mut left = pairs.len() as u32;
        let mut at = first;
        while left > 0 {
            let item = items.next().unwrap_or_else(|| panic!("{what}: too few items"));
            let count = left.min(tile);
            assert_eq!((item.expert, item.first, item.count), (expert as u32, at, count), "{what}");
            left -= count;
            at += count;
        }
        first += pairs.len() as u32;
    }
    assert!(items.next().is_none(), "{what}: too many items");
    assert_eq!(plan.first(shape.n_experts), n_valid as u32);
    for (pair, id) in picks.ids.iter().enumerate() {
        if *id >= shape.n_experts {
            assert_eq!(plan.position(pair as u32), None, "{what}: pair {pair} has no position");
        }
    }
}

/// Builds the plan of `picks` on `device` and returns its bytes.
fn build_plan(
    device: &Device,
    rows: &ExpertRowKernels,
    shape: &ExpertShape,
    picks: &Picks,
    n_rows: u32,
    tile: u32,
) -> Vec<u8> {
    let stream = Stream::new(device, 0).unwrap();
    let record = picks.record();
    let record_buf = DeviceBuffer::new(device, record.len() as u64).unwrap();
    record_buf.write(0, &record).unwrap();
    let bytes = shape.rows_plan_bytes(n_rows, tile);
    let plan = DeviceBuffer::new(device, bytes).unwrap();
    // Garbage the plan must not depend on.
    plan.write(0, &vec![0xA5u8; bytes as usize]).unwrap();
    // SAFETY: the buffers were sized for this shape and number of rows.
    unsafe {
        rows.plan_rows(&stream, shape, n_rows, tile, record_buf.ptr(), plan.ptr(), 7).unwrap()
    };
    stream.synchronize().unwrap();
    let mut out = vec![0u8; bytes as usize];
    plan.read(0, &mut out).unwrap();
    out
}

#[test]
fn the_plan_sorts_pairs_by_expert_and_cuts_items() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    // (experts, picks, rows, pattern): pairs up to 1024 take one block, more take three
    // launches.
    let cases = [
        (96u32, 4u32, 1u32, Pattern::Random),
        (96, 4, 5, Pattern::OneExpertTakesEveryRow),
        (96, 4, 24, Pattern::EveryRowDifferent),
        (96, 4, 100, Pattern::Skewed),
        (96, 4, 256, Pattern::Random),
        (512, 10, 102, Pattern::Random),
        (512, 10, 103, Pattern::Random),
        (512, 10, 700, Pattern::Skewed),
        (512, 10, 1500, Pattern::OneExpertTakesEveryRow),
        (1024, 16, 300, Pattern::Random),
        (7, 3, 400, Pattern::Random),
    ];
    for device in gpus {
        let rows = ExpertRowKernels::load(device).unwrap();
        for (n_experts, top_k, n_rows, pattern) in &cases {
            let shape = ExpertShape {
                n_embd: 512,
                n_ff: 128,
                n_experts: *n_experts,
                top_k: *top_k,
                gate_ty: GgmlType::IQ2_S,
                down_ty: GgmlType::Q2_0,
            };
            let picks = picks(pattern, &shape, *n_rows, &mut rng);
            for tile in ROWS_TILES {
                let bytes = build_plan(device, &rows, &shape, &picks, *n_rows, tile);
                let plan = RowsPlan::parse(&bytes, &shape, *n_rows, tile).unwrap();
                assert_eq!(plan.token(), 7);
                assert_eq!(plan.missing_pairs(), 0);
                let what =
                    format!("{n_experts} experts, {top_k} picks, {n_rows} rows, tile {tile}");
                check_plan(&shape, &picks, &plan, tile, &what);
            }
        }
    }
}

#[test]
fn the_plan_drops_ids_out_of_range() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(99);
    for device in gpus {
        let rows = ExpertRowKernels::load(device).unwrap();
        // 20 rows and 2000 rows: one block and several.
        for n_rows in [20u32, 700] {
            let shape = ExpertShape {
                n_embd: 512,
                n_ff: 128,
                n_experts: 50,
                top_k: 3,
                gate_ty: GgmlType::IQ2_XXS,
                down_ty: GgmlType::Q2_0,
            };
            let mut picks = picks(&Pattern::Random, &shape, n_rows, &mut rng);
            for (i, id) in picks.ids.iter_mut().enumerate() {
                if i % 7 == 3 {
                    *id = [50, 51, 1000, u32::MAX][i % 4];
                }
            }
            let bytes = build_plan(device, &rows, &shape, &picks, n_rows, 16);
            let plan = RowsPlan::parse(&bytes, &shape, n_rows, 16).unwrap();
            check_plan(&shape, &picks, &plan, 16, "ids out of range");
        }
    }
}

// ---------------------------------------------------------------------------
// The products.

/// The exact result for the rows: for each pair `w * down(silu(gate x) * up x)` summed
/// per row in f64, and each pair's hidden values.
struct Exact {
    out: Vec<f64>,
    /// `hidden[pair]`: the n_ff values `silu(gate x) * (up x)`.
    hidden: Vec<Vec<f64>>,
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn dequantized(ty: GgmlType, bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    ares_quant::dequantize(ty, bytes, &mut out).unwrap();
    out
}

fn exact(
    layer: &Layer<'_>,
    x: &[f32],
    picks: &Picks,
    n_rows: u32,
    present: &dyn Fn(u32) -> bool,
) -> Exact {
    let shape = &layer.shape;
    let (n, ff, k) = (shape.n_embd as usize, shape.n_ff as usize, shape.top_k as usize);
    let mut cache: std::collections::BTreeMap<u32, [Vec<f32>; 3]> = Default::default();
    let mut out = vec![0.0f64; n_rows as usize * n];
    let mut hidden = vec![vec![0.0f64; ff]; n_rows as usize * k];
    for row in 0..n_rows as usize {
        let xr = &x[row * n..(row + 1) * n];
        for j in 0..k {
            let pair = row * k + j;
            let e = picks.ids[pair];
            if e >= shape.n_experts || !present(e) {
                continue;
            }
            let [gate, up, down] = cache.entry(e).or_insert_with(|| {
                let [g, u, d] = (layer.expert)(e);
                [
                    dequantized(shape.gate_ty, &g, ff * n),
                    dequantized(shape.gate_ty, &u, ff * n),
                    dequantized(shape.down_ty, &d, n * ff),
                ]
            });
            let dot = |m: &[f32], len: usize, i: usize, v: &[f64]| -> f64 {
                m[i * len..(i + 1) * len].iter().zip(v).map(|(w, v)| f64::from(*w) * v).sum()
            };
            let xd: Vec<f64> = xr.iter().map(|v| f64::from(*v)).collect();
            let h: Vec<f64> =
                (0..ff).map(|i| silu(dot(gate, n, i, &xd)) * dot(up, n, i, &xd)).collect();
            let weight = f64::from(picks.weights[pair]);
            for o in 0..n {
                out[row * n + o] += weight * dot(down, ff, o, &h);
            }
            hidden[pair] = h;
        }
    }
    Exact { out, hidden }
}

/// What a run left on the device.
struct Run {
    out: Vec<f32>,
    hidden: Vec<u8>,
    plan: Vec<u8>,
}

/// Runs the layer's experts for `x` and `picks` on `device`; `present` says which
/// experts have an address.
#[allow(clippy::too_many_arguments)]
fn run(
    device: &Device,
    layer: &Layer<'_>,
    x: &[f32],
    picks: &Picks,
    n_rows: u32,
    tile: u32,
    way: Accumulate,
    present: &dyn Fn(u32) -> bool,
) -> Run {
    let shape = &layer.shape;
    let rows = ExpertRowKernels::load(device).unwrap();
    let kernels = Kernels::load(device).unwrap();
    let stream = Stream::new(device, 0).unwrap();
    let n = shape.n_embd as usize;

    // The experts that are present, in scrambled slots of a pool.
    let slot_bytes = shape.slot_bytes().unwrap();
    let n_slots = u64::from(shape.n_experts) + 3;
    let pool = DeviceBuffer::new(device, n_slots * slot_bytes).unwrap();
    // What a slot that no expert was put in holds: bytes that would give some output if
    // a kernel read them.
    pool.write(0, &vec![0x5Au8; (n_slots * slot_bytes) as usize]).unwrap();
    let mut slots: Vec<u64> = (0..n_slots).collect();
    let mut shuffle = Rng(0x1234_5678_9abc_def1 ^ u64::from(shape.n_experts));
    for i in (1..slots.len()).rev() {
        slots.swap(i, shuffle.below(i as u32 + 1) as usize);
    }
    let mut addresses = vec![0u64; shape.n_experts as usize];
    for e in 0..shape.n_experts {
        if !present(e) {
            continue;
        }
        let [gate, up, down] = (layer.expert)(e);
        let base = slots[e as usize] * slot_bytes;
        pool.write(base, &gate).unwrap();
        pool.write(base + shape.up_offset().unwrap(), &up).unwrap();
        pool.write(base + shape.down_offset().unwrap(), &down).unwrap();
        addresses[e as usize] = pool.ptr().as_raw() + base;
    }
    let table = DeviceBuffer::new(device, 8 * u64::from(shape.n_experts)).unwrap();
    table.write(0, &addresses.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>()).unwrap();

    let x_buf = DeviceBuffer::new(device, 4 * x.len() as u64).unwrap();
    x_buf.write(0, &f32_bytes(x)).unwrap();
    let act_bytes = Kernels::prefill_activation_bytes(PrefillPath::F16, shape.n_embd, n_rows);
    assert_eq!(act_bytes, shape.rows_activation_bytes(n_rows));
    let act = DeviceBuffer::new(device, act_bytes).unwrap();
    let record = picks.record();
    let record_buf = DeviceBuffer::new(device, record.len() as u64).unwrap();
    record_buf.write(0, &record).unwrap();
    let plan_bytes = shape.rows_plan_bytes(n_rows, tile);
    let plan = DeviceBuffer::new(device, plan_bytes).unwrap();
    let hidden = DeviceBuffer::new(device, shape.rows_hidden_bytes(n_rows)).unwrap();
    hidden.write(0, &vec![0xC3u8; shape.rows_hidden_bytes(n_rows) as usize]).unwrap();
    let scratch = DeviceBuffer::new(device, shape.rows_scratch_bytes(n_rows).max(16)).unwrap();
    scratch.write(0, &vec![0xC3u8; shape.rows_scratch_bytes(n_rows).max(16) as usize]).unwrap();
    let out = DeviceBuffer::new(device, 4 * u64::from(n_rows) * n as u64).unwrap();
    out.write(0, &vec![0u8; 4 * n_rows as usize * n]).unwrap();

    // SAFETY: every buffer was sized for this shape, number of rows and tile; the
    // table names slots of the pool that hold the experts.
    unsafe {
        kernels
            .prepare_prefill_activations(
                &stream,
                PrefillPath::F16,
                x_buf.ptr(),
                act.ptr(),
                shape.n_embd,
                n_rows,
            )
            .unwrap();
        rows.plan_rows(&stream, shape, n_rows, tile, record_buf.ptr(), plan.ptr(), 1).unwrap();
        rows.rows_experts(
            &stream,
            shape,
            n_rows,
            tile,
            way,
            table.ptr(),
            plan.ptr(),
            act.ptr(),
            hidden.ptr(),
            scratch.ptr(),
            out.ptr(),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let mut out_bytes = vec![0u8; 4 * n_rows as usize * n];
    out.read(0, &mut out_bytes).unwrap();
    let mut hidden_bytes = vec![0u8; shape.rows_hidden_bytes(n_rows) as usize];
    hidden.read(0, &mut hidden_bytes).unwrap();
    let mut plan_out = vec![0u8; plan_bytes as usize];
    plan.read(0, &mut plan_out).unwrap();
    Run { out: bytes_f32(&out_bytes), hidden: hidden_bytes, plan: plan_out }
}

/// The hidden values of the pair at sorted position `pos` as the device left them, in
/// the order of the matrix rows.
fn device_hidden(shape: &ExpertShape, n_rows: u32, run: &Run, pos: usize) -> Vec<f64> {
    let (ff, pairs) = (shape.n_ff as usize, (n_rows * shape.top_k) as usize);
    let scales_at = (2 * ff * pairs).next_multiple_of(16);
    let value = |i: usize| -> f64 {
        let at = 2 * (pos * ff + i);
        f64::from(ares_quant::f16_to_f32(u16::from_le_bytes([run.hidden[at], run.hidden[at + 1]])))
    };
    let scale = |tile: usize| -> f64 {
        let at = scales_at + 4 * (tile * pairs + pos);
        f64::from(f32::from_le_bytes(run.hidden[at..at + 4].try_into().unwrap()))
    };
    // Within each 16 values the device stores row r and row r + 8 side by side.
    (0..ff)
        .map(|j| {
            let (group, r) = (j / 16, j % 16);
            let at = group * 16 + 2 * (r % 8) + r / 8;
            value(at) * scale(j / 64)
        })
        .collect()
}

struct Deviation {
    /// rms error of the output relative to the rms of the exact output.
    out: f64,
    /// The same for the hidden values.
    hidden: f64,
}

/// rms of the difference over the rms of the exact values.
fn relative_rms(pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (got, want) in pairs {
        err += (got - want).powi(2);
        norm += want * want;
    }
    (err / norm.max(1e-30)).sqrt()
}

/// Runs a layer and checks it against the exact arithmetic. Returns the deviation.
#[allow(clippy::too_many_arguments)]
fn check_layer(
    device: &Device,
    layer: &Layer<'_>,
    x: &[f32],
    picks: &Picks,
    n_rows: u32,
    tile: u32,
    way: Accumulate,
    present: &dyn Fn(u32) -> bool,
) -> Deviation {
    let exact = exact(layer, x, picks, n_rows, present);
    check_layer_against(device, layer, x, picks, n_rows, tile, way, present, &exact)
}

/// [`check_layer`] against an exact result the caller has (the same on every device).
#[allow(clippy::too_many_arguments)]
fn check_layer_against(
    device: &Device,
    layer: &Layer<'_>,
    x: &[f32],
    picks: &Picks,
    n_rows: u32,
    tile: u32,
    way: Accumulate,
    present: &dyn Fn(u32) -> bool,
    exact: &Exact,
) -> Deviation {
    let shape = &layer.shape;
    let run = run(device, layer, x, picks, n_rows, tile, way, present);
    let plan = RowsPlan::parse(&run.plan, shape, n_rows, tile).expect("a plan");
    let out = relative_rms(run.out.iter().zip(&exact.out).map(|(g, e)| (f64::from(*g), *e)));

    // The hidden values of the pairs whose expert was present.
    let mut hidden_pairs = Vec::new();
    for (pair, id) in picks.ids.iter().enumerate() {
        if *id >= shape.n_experts || !present(*id) {
            continue;
        }
        let pos = plan.position(pair as u32).unwrap() as usize;
        let got = device_hidden(shape, n_rows, &run, pos);
        hidden_pairs.extend(got.into_iter().zip(exact.hidden[pair].iter().copied()));
    }
    let dev = Deviation { out, hidden: relative_rms(hidden_pairs.into_iter()) };
    eprintln!(
        "{} on {}: {n_rows} rows, tile {tile}, {way:?}: hidden {:.2e}, output {:.2e} rms",
        layer.name,
        device.info().name,
        dev.hidden,
        dev.out
    );
    dev
}

fn activations(rng: &mut Rng, n_rows: u32, n: u32) -> Vec<f32> {
    (0..n_rows * n).map(|_| rng.activation()).collect()
}

#[test]
fn experts_match_the_reference_on_generated_blocks() {
    let Some(gpus) = gpus() else { return };
    for (i, gate_ty) in
        [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M].into_iter().enumerate()
    {
        let layer = synthetic_layer(gate_ty, 40, 4, 100 + i as u64);
        let mut rng = Rng(0x9e3779b97f4a7c15 + i as u64);
        let n = layer.shape.n_embd;
        for (n_rows, tile, way, pattern) in [
            (1u32, 8u32, Accumulate::Ordered, Pattern::Random),
            (3, 8, Accumulate::Atomic, Pattern::Random),
            (9, 16, Accumulate::Ordered, Pattern::OneExpertTakesEveryRow),
            (30, 32, Accumulate::Atomic, Pattern::EveryRowDifferent),
            (100, 64, Accumulate::Ordered, Pattern::Skewed),
            (100, 8, Accumulate::Ordered, Pattern::Random),
        ] {
            let x = activations(&mut rng, n_rows, n);
            let picks = picks(&pattern, &layer.shape, n_rows, &mut rng);
            for device in gpus {
                let dev = check_layer(device, &layer, &x, &picks, n_rows, tile, way, &|_| true);
                assert!(dev.hidden <= 2e-3, "hidden values off by {}", dev.hidden);
                assert!(dev.out <= 2e-3, "output off by {}", dev.out);
            }
        }
    }
}

/// Every weight type as the type of gate and up (with down as Q2_0) and as the type of
/// down (with gate and up as IQ2_S), and a few pairs of other types, on every tile and
/// both sums: against the exact result, within the bound of the types the kernels were
/// written for.
#[test]
fn experts_of_every_weight_type_match_the_reference() {
    let Some(gpus) = gpus() else { return };
    let mut pairs: Vec<(GgmlType, GgmlType)> = Vec::new();
    for (ty, _) in TYPES {
        pairs.push((ty, GgmlType::Q2_0));
        pairs.push((GgmlType::IQ2_S, ty));
    }
    pairs.extend([
        (GgmlType::Q4_K, GgmlType::Q5_K),
        (GgmlType::F16, GgmlType::BF16),
        (GgmlType::Q8_0, GgmlType::Q6_K),
        (GgmlType::TQ1_0, GgmlType::IQ1_S),
    ]);
    let steps = [
        (1u32, 8u32, Accumulate::Ordered, Pattern::Random),
        (9, 16, Accumulate::Atomic, Pattern::OneExpertTakesEveryRow),
        (30, 32, Accumulate::Atomic, Pattern::EveryRowDifferent),
        (70, 64, Accumulate::Ordered, Pattern::Skewed),
    ];
    for (i, (gate_ty, down_ty)) in pairs.into_iter().enumerate() {
        let layer = synthetic_layer_of(gate_ty, down_ty, 24, 4, 500 + i as u64);
        check_layer_of_types(gpus, &layer, 0x9e3779b97f4a7c15 + i as u64, &steps);
    }
}

/// Runs `layer` with each of `steps` (rows, tile, sum, picks) on every device and holds the
/// hidden values and the output to the bound of the types the kernels were written for.
fn check_layer_of_types(
    gpus: &[Device],
    layer: &Layer<'_>,
    seed: u64,
    steps: &[(u32, u32, Accumulate, Pattern)],
) {
    layer.shape.check_rows().expect("a shape the kernels take");
    let mut rng = Rng(seed);
    let n = layer.shape.n_embd;
    let mut worst = (0.0f64, 0.0f64);
    for (n_rows, tile, way, pattern) in steps {
        let (n_rows, tile, way) = (*n_rows, *tile, *way);
        let x = activations(&mut rng, n_rows, n);
        let picks = picks(pattern, &layer.shape, n_rows, &mut rng);
        let exact = exact(layer, &x, &picks, n_rows, &|_| true);
        for device in gpus {
            let dev = check_layer_against(
                device,
                layer,
                &x,
                &picks,
                n_rows,
                tile,
                way,
                &|_| true,
                &exact,
            );
            // A comparison with what is not a number is false either way: say so first.
            assert!(
                dev.hidden.is_finite() && dev.out.is_finite(),
                "{}: {n_rows} rows, tile {tile}: not a number",
                layer.name
            );
            assert!(dev.hidden <= 2e-3, "{}: hidden values off by {}", layer.name, dev.hidden);
            assert!(dev.out <= 2e-3, "{}: output off by {}", layer.name, dev.out);
            worst = (worst.0.max(dev.hidden), worst.1.max(dev.out));
        }
    }
    eprintln!("ERR {}: hidden {:.2e}, output {:.2e}", layer.name, worst.0, worst.1);
}

/// The widths of an expert of the model the kernels were written for: rows of 2560 values
/// for gate and up, and of 640 for down, or 768 for a type of down whose blocks of 256
/// values do not divide 640.
fn wide_shape(gate_ty: GgmlType, down_ty: GgmlType) -> ExpertShape {
    let n_ff = if 640 % down_ty.block_values() == 0 { 640 } else { 768 };
    ExpertShape { n_embd: 2560, n_ff, n_experts: 8, top_k: 2, gate_ty, down_ty }
}

/// The steps of the tests at real widths: every tile, both sums, an expert with more pairs
/// than the largest tile holds.
fn wide_steps() -> [(u32, u32, Accumulate, Pattern); 4] {
    [
        (1, 8, Accumulate::Ordered, Pattern::Random),
        (9, 16, Accumulate::Atomic, Pattern::OneExpertTakesEveryRow),
        (17, 32, Accumulate::Atomic, Pattern::Skewed),
        (65, 64, Accumulate::Ordered, Pattern::OneExpertTakesEveryRow),
    ]
}

/// Every weight type at the widths of a real expert, as the type of gate and up and as the
/// type of down: rows of 20 steps of the gate and up kernel and 10 or 12 of the down
/// kernel, and row lengths in bytes that are not multiples of 16 for most types.
#[test]
fn experts_of_every_weight_type_match_the_reference_at_real_widths() {
    let Some(gpus) = gpus() else { return };
    for (i, (ty, _)) in TYPES.into_iter().enumerate() {
        for (gate_ty, down_ty) in [(ty, GgmlType::Q2_0), (GgmlType::IQ2_S, ty)] {
            let shape = wide_shape(gate_ty, down_ty);
            let name = format!("{gate_ty} and {down_ty}, {} x {}", shape.n_embd, shape.n_ff);
            let layer = layer_of(name, shape, 700 + i as u64, &mut random_blocks);
            check_layer_of_types(gpus, &layer, 0x2545f4914f6cdd1d + i as u64, &wide_steps());
        }
    }
}

/// The largest tensor of two or more dimensions of every weight type the file has: the
/// blocks a quantizer wrote, which the tests take as the matrices of experts.
fn real_blocks(gguf: &ares_gguf::Gguf) -> Vec<(GgmlType, &[u8])> {
    let mut found: Vec<(GgmlType, &[u8])> = Vec::new();
    for tensor in gguf.tensors() {
        if tensor.dims.as_slice().len() < 2 || !TYPES.iter().any(|(ty, _)| *ty == tensor.ty) {
            continue;
        }
        let data = gguf.data(tensor);
        match found.iter_mut().find(|(ty, _)| *ty == tensor.ty) {
            Some((_, largest)) if largest.len() < data.len() => *largest = data,
            Some(_) => {}
            None => found.push((tensor.ty, data)),
        }
    }
    found
}

/// A source of blocks for [`layer_of`] that takes the blocks of `ty` from `data`, one
/// stretch after the other and from the start again at the end, and generates the blocks
/// of every other type.
fn blocks_of<'a>(
    ty: GgmlType,
    data: &'a [u8],
) -> impl FnMut(&mut Rng, GgmlType, usize) -> Vec<u8> + 'a {
    let block = ty.block_bytes() as usize;
    let mut at = 0;
    move |rng, wanted, n_blocks| {
        if wanted != ty {
            return random_blocks(rng, wanted, n_blocks);
        }
        let mut bytes = Vec::with_capacity(n_blocks * block);
        while bytes.len() < n_blocks * block {
            if at + block > data.len() {
                at = 0;
            }
            let take = (n_blocks * block - bytes.len()).min((data.len() - at) / block * block);
            bytes.extend_from_slice(&data[at..at + take]);
            at += take;
        }
        bytes
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
            check_layer_of_types(gpus, &layer, 0x853c49e6748fea9b + i as u64, &wide_steps());
        }
    }
}

#[test]
fn missing_experts_are_reported_and_not_read() {
    let Some(gpus) = gpus() else { return };
    let layer = synthetic_layer(GgmlType::IQ2_S, 40, 4, 7);
    let shape = &layer.shape;
    let mut rng = Rng(77);
    // Every third expert has no address.
    let present = |e: u32| !e.is_multiple_of(3);
    for (n_rows, tile) in [(1u32, 8u32), (10, 16), (150, 64)] {
        let x = activations(&mut rng, n_rows, shape.n_embd);
        let picks = picks(&Pattern::Random, shape, n_rows, &mut rng);
        let missing_pairs = picks.ids.iter().filter(|e| !present(**e)).count() as u32;
        assert!(missing_pairs > 0);
        for device in gpus {
            for way in [Accumulate::Ordered, Accumulate::Atomic] {
                let dev = check_layer(device, &layer, &x, &picks, n_rows, tile, way, &present);
                assert!(dev.out <= 2e-3, "the output of the present experts is off by {}", dev.out);
                assert!(dev.hidden <= 2e-3);
                let run = run(device, &layer, &x, &picks, n_rows, tile, way, &present);
                let plan = RowsPlan::parse(&run.plan, shape, n_rows, tile).unwrap();
                assert_eq!(plan.missing_pairs(), missing_pairs);
                for e in 0..shape.n_experts {
                    let picked = picks.ids.contains(&e);
                    assert_eq!(plan.was_missing(e), picked && !present(e), "expert {e}");
                }
            }
        }
    }
}

#[test]
fn a_row_gets_the_same_bits_alone_and_in_a_batch() {
    let Some(gpus) = gpus() else { return };
    for gate_ty in [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M] {
        let layer = synthetic_layer(gate_ty, 40, 4, 21);
        let shape = &layer.shape;
        let n = shape.n_embd as usize;
        let mut rng = Rng(99);
        let n_rows = 70u32;
        let x = activations(&mut rng, n_rows, shape.n_embd);
        let all = picks(&Pattern::Skewed, shape, n_rows, &mut rng);
        let k = shape.top_k as usize;
        for device in gpus {
            let batch = run(device, &layer, &x, &all, n_rows, 64, Accumulate::Ordered, &|_| true);
            for row in [0usize, 3, 31, 69] {
                let one = Picks {
                    ids: all.ids[row * k..(row + 1) * k].to_vec(),
                    weights: all.weights[row * k..(row + 1) * k].to_vec(),
                };
                let alone = run(
                    device,
                    &layer,
                    &x[row * n..(row + 1) * n],
                    &one,
                    1,
                    8,
                    Accumulate::Ordered,
                    &|_| true,
                );
                assert_eq!(alone.out, batch.out[row * n..(row + 1) * n], "{gate_ty}: row {row}");
            }
            // The rows in another order give each row the same bits.
            let order: Vec<usize> = (0..n_rows as usize).rev().collect();
            let xr: Vec<f32> = order.iter().flat_map(|r| x[r * n..(r + 1) * n].to_vec()).collect();
            let shuffled = Picks {
                ids: order.iter().flat_map(|r| all.ids[r * k..(r + 1) * k].to_vec()).collect(),
                weights: order
                    .iter()
                    .flat_map(|r| all.weights[r * k..(r + 1) * k].to_vec())
                    .collect(),
            };
            let reversed =
                run(device, &layer, &xr, &shuffled, n_rows, 32, Accumulate::Ordered, &|_| true);
            for (i, r) in order.iter().enumerate() {
                assert_eq!(
                    reversed.out[i * n..(i + 1) * n],
                    batch.out[r * n..(r + 1) * n],
                    "{gate_ty}: row {r} in another place"
                );
            }
        }
    }
}

#[test]
fn rows_of_any_magnitude_keep_their_precision() {
    let Some(gpus) = gpus() else { return };
    let layer = synthetic_layer(GgmlType::IQ2_XXS, 40, 4, 5);
    let shape = &layer.shape;
    let n = shape.n_embd as usize;
    let mut rng = Rng(3);
    let n_rows = 12u32;
    let mut x = activations(&mut rng, n_rows, shape.n_embd);
    for (row, factor) in [1e-4f32, 1e-2, 1.0, 1e2, 1e4, 1e5].iter().cycle().take(12).enumerate() {
        x[row * n..(row + 1) * n].iter_mut().for_each(|v| *v *= factor);
    }
    let picks = picks(&Pattern::Random, shape, n_rows, &mut rng);
    for device in gpus {
        let run = run(device, &layer, &x, &picks, n_rows, 16, Accumulate::Ordered, &|_| true);
        let exact = exact(&layer, &x, &picks, n_rows, &|_| true);
        for row in 0..n_rows as usize {
            let dev = relative_rms(
                run.out[row * n..(row + 1) * n]
                    .iter()
                    .zip(&exact.out[row * n..(row + 1) * n])
                    .map(|(g, e)| (f64::from(*g), *e)),
            );
            assert!(dev <= 2e-3, "row {row}: off by {dev}");
        }
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
    let mut layers: std::collections::BTreeMap<String, usize> = Default::default();
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
        if shape.check().is_err() {
            eprintln!("blk.{index}: no many-row kernel for {} / {}", gate.ty, down.ty);
            continue;
        }
        let (gate_bytes, down_bytes) =
            (shape.gate_bytes().unwrap() as usize, shape.down_bytes().unwrap() as usize);
        let gguf = &gguf;
        let layer = Layer {
            name: format!("blk.{index} ({} / {})", gate.ty, down.ty),
            shape,
            expert: Box::new(move |e| {
                let e = e as usize;
                [
                    gguf.data(gate)[e * gate_bytes..(e + 1) * gate_bytes].to_vec(),
                    gguf.data(up)[e * gate_bytes..(e + 1) * gate_bytes].to_vec(),
                    gguf.data(down)[e * down_bytes..(e + 1) * down_bytes].to_vec(),
                ]
            }),
        };
        for (n_rows, tile, way) in [
            (1u32, 8u32, Accumulate::Ordered),
            (5, 16, Accumulate::Atomic),
            (12, 8, Accumulate::Ordered),
        ] {
            let x = activations(&mut rng, n_rows, shape.n_embd);
            let picks = picks(&Pattern::Random, &shape, n_rows, &mut rng);
            // Only the experts the rows pick are put on the device.
            let present = |e: u32| picks.ids.contains(&e);
            for device in gpus {
                let dev = check_layer(device, &layer, &x, &picks, n_rows, tile, way, &present);
                assert!(dev.hidden <= 2e-3, "hidden values off by {}", dev.hidden);
                assert!(dev.out <= 2e-3, "output off by {}", dev.out);
            }
        }
    }
}

#[test]
fn the_break_even_with_the_decode_kernels_is_measured_on_the_device() {
    let Some(gpus) = gpus() else { return };
    for gate_ty in [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M] {
        let shape = ExpertShape {
            n_embd: 2560,
            n_ff: 640,
            n_experts: 512,
            top_k: 10,
            gate_ty,
            down_ty: GgmlType::Q2_0,
        };
        for device in gpus {
            let rows = ExpertRowKernels::load(device).unwrap();
            let decode = ares_cuda::ExpertKernels::load(device).unwrap();
            let up_to = rows.few_rows_break_even(&decode, &shape).unwrap();
            eprintln!(
                "{gate_ty} on {}: the decode kernels are faster up to {up_to} rows",
                device.info().name
            );
            assert!(up_to <= ares_cuda::MAX_BATCH);
        }
    }
}
