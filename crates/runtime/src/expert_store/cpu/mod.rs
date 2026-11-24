//! Experts on the CPU: the one place where the engine computes on it.
//!
//! An expert that a step needs and that is not in a VRAM slot is either
//! streamed to the device or computed here, from its bytes in the RAM tier,
//! whichever the measured cost model of the store says is sooner. This
//! module is the computing half:
//!
//! - the kernels (`kernels.rs`): gate and up of an expert on the same
//!   input, `silu(gate) * up`, down, for the rows of a step that picked the
//!   expert, straight on the file's quantized bytes (kernels written for
//!   IQ2_S, IQ2_XXS, IQ1_M, IQ3_XXS, IQ4_XS, Q4_K, Q5_K and Q2_0 gate and up and for Q2_0,
//!   Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and IQ4_NL down, and every other weight
//!   type through the reference decoder). The arithmetic is the device's: the
//!   same int8 activations, exact integer sums per 32 values, the hidden
//!   values quantized the same way. A row therefore loses nothing when the
//!   CPU computes one of its experts.
//! - a [`LayerJob`]: the experts of one layer that the CPU computes for one
//!   step, cut into tasks that any thread can run, and their weighted sum
//!   per row, which is what the device adds to the rows' outputs.
//! - the [`CpuPool`] (`pool.rs`): the threads that run the tasks, and where
//!   they sit on a CPU with cores of two kinds. Workers only compute; they
//!   own no state of the store.
//!
//! The written kernels need x86-64-v3 (AVX2, FMA, F16C, BMI2), and their
//! down products have a second form for AVX-VNNI; [`Features::detect`] says
//! once, when the pool is built, what the CPU has. A CPU without x86-64-v3
//! computes every type through the reference decoder with portable code (the
//! decoded path of `kernels.rs`), which is slower and gives the same sums.

mod kernels;
mod pool;
mod tables;
#[cfg(test)]
mod test_rig;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_cuda::ExpertShape;
use ares_quant::act;

use kernels::{DownExpert, DownKind, GateKind, Hidden, Order, Planes, RowIn};
pub use pool::{CpuPool, Placement, Topology};
use tables::Lanes16;

/// The most rows of a step that can share an expert in one job: the rows of
/// a decode step of the device kernels.
pub const MAX_ROWS: usize = 8;

/// Hidden values per gate and up task, and output values per down task.
const TILE: usize = 32;
const DOWN_ROWS: usize = 64;

/// Why the CPU cannot compute experts.
#[derive(Debug, thiserror::Error)]
pub enum CpuError {
    #[error("no CPU kernel for experts with {what} of type {ty}")]
    Type { what: &'static str, ty: GgmlType },
    #[error("expert job: {0}")]
    Job(String),
}

/// What the CPU kernels may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    /// x86-64-v3 (AVX2, FMA, F16C, BMI2): the written kernels. Without it
    /// every type goes through the portable decoded path.
    pub vector: bool,
    /// AVX-VNNI: the down product sums four byte products in one instruction.
    pub vnni: bool,
    /// How codebook entries are fetched.
    pub decode: Decode,
}

/// How the gate and up kernels fetch the four codebook entries of 32
/// weights. Which is faster depends on the core: a gather is slow on the
/// efficiency cores of a hybrid CPU, where four loads are 2 to 4 times as
/// fast, and the two are close on the performance cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decode {
    /// Whichever the thread that runs a task measured as faster on its core.
    Fastest,
    Gather,
    Loads,
}

impl Features {
    /// What this CPU has. This is the only place it is asked: a [`LayerJob`]
    /// can only be made with the value it returns (or one with less), so no
    /// kernel runs on a CPU without the instructions it uses.
    pub fn detect() -> Self {
        let vector = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("bmi2");
        Self {
            vector,
            vnni: vector && is_x86_feature_detected!("avxvnni"),
            decode: Decode::Fastest,
        }
    }

    /// The same CPU without AVX-VNNI, to measure what it gives.
    pub fn without_vnni(self) -> Self {
        Self { vnni: false, ..self }
    }

    /// The same CPU with the portable code only: what a CPU without
    /// x86-64-v3 runs, to test and measure it here.
    pub fn portable(self) -> Self {
        Self { vector: false, vnni: false, ..self }
    }

    /// The same CPU with one way of fetching codebook entries on every
    /// thread, to measure the two against each other.
    pub fn with_decode(self, decode: Decode) -> Self {
        Self { decode, ..self }
    }
}

/// One expert of a job: where its three matrices are, in the file's bytes,
/// and which rows of the step use it with what weight.
#[derive(Clone, Copy, Debug)]
pub struct CpuExpert {
    pub gate: *const u8,
    pub up: *const u8,
    pub down: *const u8,
    /// Bit `b` is set when row `b` of the step picked the expert.
    pub rows: u32,
    /// Row `b`'s weight for the expert.
    pub weight: [f32; MAX_ROWS],
}

/// The gate types a thread measures itself in: those with a kernel written
/// for them (IQ2_S, IQ2_XXS, IQ1_M, IQ3_XXS, IQ4_XS, Q4_K, Q5_K, Q2_0).
pub const MEASURED: usize = GateKind::WRITTEN.len();

/// What a thread measured of itself, once: which way of fetching codebook
/// entries is faster on its core for each gate type, and how long 32 hidden
/// values of one row take it that way, in nanoseconds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThreadSpeed {
    pub(crate) gather: [bool; MEASURED],
    pub(crate) tile_ns: [u32; MEASURED],
}

/// The speeds of the threads that share jobs: what a slow thread needs to
/// know to leave the last pieces of a job to the fast ones.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Team {
    /// The tile time of the fastest worker, per gate type.
    fast_tile_ns: [u32; MEASURED],
    /// How many workers are within half of that time again.
    fast: [u32; MEASURED],
}

impl Team {
    pub(crate) fn of(speeds: &[ThreadSpeed]) -> Self {
        let mut team = Self { fast_tile_ns: [u32::MAX; MEASURED], fast: [0; MEASURED] };
        for kind in 0..MEASURED {
            let best = speeds.iter().map(|s| s.tile_ns[kind]).min().unwrap_or(u32::MAX);
            team.fast_tile_ns[kind] = best;
            team.fast[kind] =
                speeds.iter().filter(|s| !is_slow(s.tile_ns[kind], best)).count() as u32;
        }
        team
    }
}

/// Whether a thread with this tile time counts as slow next to the fastest.
fn is_slow(tile_ns: u32, fastest_ns: u32) -> bool {
    u64::from(tile_ns) * 2 > u64::from(fastest_ns) * 3
}

thread_local! {
    static SPEED: std::cell::Cell<Option<ThreadSpeed>> = const { std::cell::Cell::new(None) };
}

/// This thread's speed, measured at its first call: a tile of each gate
/// type with a kernel of its own, both ways where it fetches codebook
/// entries, on bytes that vary. A thread bound to a core measures that core.
/// Without the written kernels (`vector` false) there is nothing to choose
/// and every thread counts as fast.
pub(crate) fn thread_speed(vector: bool) -> ThreadSpeed {
    if !vector {
        return ThreadSpeed { gather: [false; MEASURED], tile_ns: [1; MEASURED] };
    }
    if let Some(speed) = SPEED.get() {
        return speed;
    }
    let tables = tables::tables();
    let n_in = 2560;
    let x: Vec<f32> = (0..n_in).map(|i| ((i * 37 % 255) as f32 - 127.0) / 64.0).collect();
    let act = quantize_activations(&x, n_in);
    let mut sum8 = vec![Lanes16([0; 16]); n_in / 32];
    let q: Vec<i8> = act[..n_in].iter().map(|b| *b as i8).collect();
    kernels::sum_groups(&q, &mut sum8);
    let ds_at = n_in.next_multiple_of(16);
    let planes = Planes::new(&q, &act[ds_at..], n_in);
    let row = RowIn {
        q: q.as_ptr(),
        ds: act[ds_at..].as_ptr(),
        sum8: sum8.as_ptr(),
        planes: planes.row(0),
    };
    let (mut hq, mut hsum, mut hscale) = ([0i8; 128], [0i32; 8], [0.0f32; 8]);
    let hidden = Hidden { q: hq.as_mut_ptr(), sum: hsum.as_mut_ptr(), scale: hscale.as_mut_ptr() };
    let mut speed = ThreadSpeed { gather: [true; MEASURED], tile_ns: [u32::MAX; MEASURED] };
    for (k, kind) in GateKind::WRITTEN.into_iter().enumerate() {
        let bytes = 32 * (n_in / kind.ty().block_values() as usize) * kind.block_bytes();
        // Every byte below 0x40: the scales stay small and finite.
        let fill = |seed: usize| -> Vec<u8> {
            (0..bytes).map(|i| ((i * 31 + seed * 17) ^ (i >> 7)) as u8 & 0x3f).collect()
        };
        let (gate, up) = (fill(1), fill(2));
        let ways: &[bool] = if kind.has_codebook() { &[true, false] } else { &[true] };
        for &gather in ways {
            let mut best = u32::MAX;
            for round in 0..8 {
                let start = Instant::now();
                // SAFETY: the caller found x86-64-v3 (`vector`); the matrices are 32
                // rows of `n_in` values, the row and the hidden arrays are
                // sized for them, and all of it is this thread's own.
                unsafe {
                    kernels::gate_up_tile(
                        tables,
                        kind,
                        gather,
                        false,
                        gate.as_ptr(),
                        up.as_ptr(),
                        n_in,
                        0,
                        &[row],
                        &[hidden],
                        Order::Values,
                    );
                }
                if round >= 3 {
                    best = best.min(start.elapsed().as_nanos() as u32);
                }
            }
            if best < speed.tile_ns[k] {
                speed.tile_ns[k] = best;
                speed.gather[k] = gather;
            }
        }
    }
    SPEED.set(Some(speed));
    speed
}

/// Memory that the tasks of a job write in disjoint parts.
struct Scratch<T> {
    ptr: *mut T,
    len: usize,
}

impl<T: Clone> Scratch<T> {
    fn new(len: usize, value: T) -> Self {
        let boxed = vec![value; len].into_boxed_slice();
        Self { ptr: Box::into_raw(boxed).cast(), len }
    }
}

impl<T> Drop for Scratch<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer and length are those of the boxed slice that
        // `new` leaked, and nothing else frees it.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ptr, self.len)) });
    }
}

#[derive(Debug)]
struct JobExpert {
    gate: *const u8,
    up: *const u8,
    down: *const u8,
    /// The pairs of the expert are `first_pair ..`, one per row in `row`: the rows of the
    /// job's activations they read (at most [`MAX_ROWS`]), and their places among the
    /// expert's pairs, by which the down kernels keep their sums.
    first_pair: usize,
    row: Vec<u32>,
    place: Vec<u8>,
    weight: Vec<f32>,
}

/// One expert of a job of any number of rows ([`LayerJob::with_rows`]): where its three
/// matrices are, in the file's bytes, and the rows of the job that use it (at most
/// [`MAX_ROWS`], each once) with their weights.
#[derive(Clone, Debug)]
pub struct RowsExpert {
    pub gate: *const u8,
    pub up: *const u8,
    pub down: *const u8,
    pub rows: Vec<(u32, f32)>,
}

/// A task of a job.
#[derive(Clone, Copy, Debug)]
enum Task {
    /// Hidden values `32 tile ..` of expert `expert` for the rows that use it.
    GateUp { expert: u32, tile: u32 },
    /// Output values `64 chunk ..` of expert `expert`'s down product for the
    /// rows that use it, added to the job's sums after those of the experts
    /// before it.
    Down { expert: u32, chunk: u32 },
}

/// The experts of one layer that the CPU computes for one step.
///
/// The job owns a copy of the rows' activations and everything it computes;
/// the weights it only points at. Its work is a list of tasks that threads
/// claim one by one, so however many take part, each task runs once and the
/// result does not depend on who ran it: the gate and up product of every
/// expert in pieces of 32 hidden values, and its down product in pieces of 64
/// output values, which start when the expert's hidden values are complete.
/// The down products of an expert come after the gate and up pieces of the
/// next one: so the reads of down, which the memory bounds, run beside the
/// decoding of gate and up, which the cores bound. The pieces of a chunk of
/// the output are added up in the order of the experts.
pub struct LayerJob {
    kind: GateKind,
    down: DownKind,
    features: Features,
    n_in: usize,
    n_ff: usize,
    n_out: usize,
    n_rows: usize,
    experts: Vec<JobExpert>,
    q: Vec<i8>,
    ds: Vec<u8>,
    sum8: Vec<Lanes16>,
    /// For a gate type of Q2_0: the activations in plane order.
    planes: Planes,
    /// The memory of the hidden values that `hiddens` point into: the codes,
    /// their sums and scales of every 16 values, `n_ff` rounded up to 128 per
    /// pair. It is only held here.
    _hidden_memory: (Scratch<i8>, Scratch<i32>, Scratch<f32>),
    out: Scratch<f32>,
    /// The hidden values of every pair, as the down kernels read them.
    hiddens: Vec<Hidden>,
    /// The tasks in the order they are claimed.
    schedule: Vec<Task>,
    /// Per expert, the gate and up tasks that have not ended.
    tiles_left: Vec<AtomicUsize>,
    /// Per chunk of the output, how many experts have added their part.
    chunk_parts: Vec<AtomicUsize>,
    /// The place in `schedule` after the last gate and up task of the last
    /// expert: the pieces a slow thread leaves to the fast ones.
    last_gate_up_end: usize,
    gate_up_tasks: usize,
    tasks: usize,
    next: AtomicUsize,
    gate_up_done: AtomicUsize,
    /// Nanoseconds from `created` to the end of the last gate and up task.
    gate_up_finished: AtomicU64,
    done: AtomicUsize,
    created: Instant,
    /// Nanoseconds from `created` to the end of the last task, plus one;
    /// 0 while the job runs.
    finished: AtomicU64,
    /// Nanoseconds threads spent in tasks.
    busy: AtomicU64,
}

// SAFETY: the raw pointers are either the job's own allocations, written by
// tasks in disjoint parts (each task index is claimed once), or weights that
// the creator guarantees to stay valid and unchanged while the job lives.
unsafe impl Send for LayerJob {}
// SAFETY: as above; the shared state is atomics.
unsafe impl Sync for LayerJob {}

impl std::fmt::Debug for LayerJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerJob")
            .field("experts", &self.experts.len())
            .field("rows", &self.n_rows)
            .field("tasks", &self.tasks)
            .field("done", &self.done.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl LayerJob {
    /// A job for `experts` of a layer of `shape` and the `n_rows` rows of a
    /// step whose activations are `act`: the bytes `ares_quantize_act` wrote
    /// for them, of which the job reads the int8 codes and the `(d, s)`
    /// pairs (the first [`act_bytes`] bytes).
    ///
    /// # Safety
    ///
    /// Every expert's `gate`, `up` and `down` must address a whole matrix of
    /// the shape (in the file's bytes), and those bytes must stay valid and
    /// unchanged until the job is dropped.
    pub unsafe fn new(
        features: Features,
        shape: &ExpertShape,
        n_rows: usize,
        act: &[u8],
        experts: &[CpuExpert],
    ) -> Result<Self, CpuError> {
        let (kind, down, q, ds, sum8, planes) = Self::inputs(features, shape, n_rows, act, true)?;
        let mut list = Vec::with_capacity(experts.len());
        for expert in experts {
            let rows: Vec<u32> = (0..n_rows as u32).filter(|b| expert.rows >> b & 1 != 0).collect();
            if rows.is_empty() || expert.rows >> n_rows != 0 {
                return Err(CpuError::Job(format!(
                    "an expert with the row mask {:#x} in a step of {n_rows} rows",
                    expert.rows
                )));
            }
            list.push(JobExpert {
                gate: expert.gate,
                up: expert.up,
                down: expert.down,
                first_pair: 0,
                place: (0..rows.len() as u8).collect(),
                weight: rows.iter().map(|b| expert.weight[*b as usize]).collect(),
                row: rows,
            });
        }
        // SAFETY: the caller's guarantee.
        unsafe { Self::build(kind, down, features, shape, n_rows, q, ds, sum8, planes, list) }
    }

    /// A job for `experts` of a layer of `shape` and the `n_rows` rows whose activations are
    /// `act` (as [`LayerJob::new`] reads them), of any number: each expert reads at most
    /// [`MAX_ROWS`] of them, so an expert that more rows picked is given as several, each
    /// with some of its rows (and read once for each). What a step of many rows misses that
    /// few of its rows picked (`hybrid/wide.rs`).
    ///
    /// # Safety
    ///
    /// As [`LayerJob::new`].
    pub unsafe fn with_rows(
        features: Features,
        shape: &ExpertShape,
        n_rows: usize,
        act: &[u8],
        experts: &[RowsExpert],
    ) -> Result<Self, CpuError> {
        let (kind, down, q, ds, sum8, planes) = Self::inputs(features, shape, n_rows, act, false)?;
        let mut list = Vec::with_capacity(experts.len());
        for expert in experts {
            let bad = expert.rows.is_empty()
                || expert.rows.len() > MAX_ROWS
                || expert.rows.iter().any(|(row, _)| *row as usize >= n_rows);
            if bad {
                return Err(CpuError::Job(format!(
                    "an expert with the rows {:?} in a job of {n_rows} rows",
                    expert.rows
                )));
            }
            list.push(JobExpert {
                gate: expert.gate,
                up: expert.up,
                down: expert.down,
                first_pair: 0,
                place: (0..expert.rows.len() as u8).collect(),
                weight: expert.rows.iter().map(|(_, w)| *w).collect(),
                row: expert.rows.iter().map(|(row, _)| *row).collect(),
            });
        }
        // SAFETY: the caller's guarantee.
        unsafe { Self::build(kind, down, features, shape, n_rows, q, ds, sum8, planes, list) }
    }

    /// What a job reads of `act`, the activations of `n_rows` rows: the kinds of its
    /// matrices, the int8 codes, their scales, the sums of IQ1_M's groups and the planes of
    /// Q2_0's. `decode` holds the rows to a decode step's.
    #[allow(clippy::type_complexity)]
    fn inputs(
        features: Features,
        shape: &ExpertShape,
        n_rows: usize,
        act: &[u8],
        decode: bool,
    ) -> Result<(GateKind, DownKind, Vec<i8>, Vec<u8>, Vec<Lanes16>, Planes), CpuError> {
        let mut kind = GateKind::of(shape.gate_ty)
            .ok_or(CpuError::Type { what: "gate and up", ty: shape.gate_ty })?;
        let mut down = DownKind::of(shape.down_ty)
            .ok_or(CpuError::Type { what: "down", ty: shape.down_ty })?;
        if !features.vector {
            // The written kernels need x86-64-v3: the decoded path takes all.
            kind = GateKind::Other(shape.gate_ty);
            down = DownKind::Other(shape.down_ty);
        }
        // Whole blocks of both types and whole groups of 32 values.
        shape.check().map_err(|e| CpuError::Job(e.to_string()))?;
        let n_in = shape.n_embd as usize;
        if n_rows == 0 || (decode && n_rows > MAX_ROWS) {
            return Err(CpuError::Job(format!("{n_rows} rows; a job takes 1 to {MAX_ROWS}")));
        }
        if act.len() < act_bytes(n_in, n_rows) {
            return Err(CpuError::Job(format!(
                "{} bytes of activations for {n_rows} rows of {n_in}",
                act.len()
            )));
        }
        let codes = n_rows * n_in;
        let q: Vec<i8> = act[..codes].iter().map(|b| *b as i8).collect();
        let ds_at = codes.next_multiple_of(16);
        let ds = act[ds_at..ds_at + codes / 32 * 8].to_vec();
        let mut sum8 = Vec::new();
        if kind == GateKind::Iq1M {
            sum8 = vec![Lanes16([0; 16]); codes / 32];
            kernels::sum_groups(&q, &mut sum8);
        }
        let planes =
            if kind == GateKind::Q2_0 { Planes::new(&q, &ds, n_in) } else { Planes::default() };

        Ok((kind, down, q, ds, sum8, planes))
    }

    /// The job of the experts `list` (their pairs not counted yet) over the inputs that
    /// [`LayerJob::inputs`] made.
    ///
    /// # Safety
    ///
    /// As [`LayerJob::new`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn build(
        kind: GateKind,
        down: DownKind,
        features: Features,
        shape: &ExpertShape,
        n_rows: usize,
        q: Vec<i8>,
        ds: Vec<u8>,
        sum8: Vec<Lanes16>,
        planes: Planes,
        mut list: Vec<JobExpert>,
    ) -> Result<Self, CpuError> {
        let (n_in, n_ff) = (shape.n_embd as usize, shape.n_ff as usize);
        let mut pairs = 0;
        for expert in &mut list {
            expert.first_pair = pairs;
            pairs += expert.row.len();
        }
        let hidden_stride = n_ff.next_multiple_of(128);
        let (tiles, chunks) = (n_ff / TILE, n_in.div_ceil(DOWN_ROWS));
        let gate_up_tasks_of =
            |e: usize| (0..tiles).map(move |t| Task::GateUp { expert: e as u32, tile: t as u32 });
        let down_tasks_of =
            |e: usize| (0..chunks).map(move |c| Task::Down { expert: e as u32, chunk: c as u32 });
        let mut schedule = Vec::with_capacity(list.len() * (tiles + chunks));
        let mut last_gate_up_end = 0;
        for e in 0..list.len() {
            schedule.extend(gate_up_tasks_of(e));
            last_gate_up_end = schedule.len();
            if e > 0 {
                schedule.extend(down_tasks_of(e - 1));
            }
        }
        if !list.is_empty() {
            schedule.extend(down_tasks_of(list.len() - 1));
        }
        let n_experts = list.len();
        let gate_up_tasks = n_experts * tiles;
        let tasks = schedule.len();
        let hidden_q = Scratch::new(pairs * hidden_stride, 0);
        let hidden_sum = Scratch::new(pairs * hidden_stride / 16, 0);
        let hidden_scale = Scratch::new(pairs * hidden_stride / 16, 0.0);
        let hiddens = (0..pairs)
            .map(|p| {
                let at = p * hidden_stride;
                // SAFETY: a pair's part of the three arrays, inside them by
                // how they were sized.
                unsafe {
                    Hidden {
                        q: hidden_q.ptr.add(at),
                        sum: hidden_sum.ptr.add(at / 16),
                        scale: hidden_scale.ptr.add(at / 16),
                    }
                }
            })
            .collect();
        let job = Self {
            kind,
            down,
            features,
            n_in,
            n_ff,
            n_out: n_in,
            n_rows,
            experts: list,
            q,
            ds,
            sum8,
            planes,
            tiles_left: (0..n_experts).map(|_| AtomicUsize::new(tiles)).collect(),
            chunk_parts: (0..chunks).map(|_| AtomicUsize::new(0)).collect(),
            _hidden_memory: (hidden_q, hidden_sum, hidden_scale),
            out: Scratch::new(n_rows * n_in, 0.0),
            hiddens,
            schedule,
            last_gate_up_end,
            gate_up_tasks,
            tasks,
            next: AtomicUsize::new(0),
            gate_up_done: AtomicUsize::new(0),
            gate_up_finished: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            created: Instant::now(),
            finished: AtomicU64::new(0),
            busy: AtomicU64::new(0),
        };
        if job.tasks == 0 {
            job.finished.store(1, Ordering::Release);
        }
        Ok(job)
    }

    /// The same job through the reference decoder for both matrices: for the
    /// tests that hold the written kernels to it.
    #[cfg(test)]
    pub(crate) fn decode_everything(&mut self) {
        self.kind = GateKind::Other(self.kind.ty());
        self.down = DownKind::Other(self.down.ty());
    }

    /// The (expert, row) pairs of the job: its work in units of the cost model.
    pub fn pairs(&self) -> usize {
        self.experts.iter().map(|e| e.row.len()).sum()
    }

    /// The experts of the job.
    pub fn experts(&self) -> usize {
        self.experts.len()
    }

    fn hidden(&self, pair: usize) -> Hidden {
        self.hiddens[pair]
    }

    /// The index of the job's gate type in a [`ThreadSpeed`].
    fn kind_index(&self) -> usize {
        self.kind.index()
    }

    /// Runs task `index`, which the caller has claimed. A down task must
    /// not start before its expert's gate and up tasks have ended.
    fn run_task(&self, index: usize, speed: &ThreadSpeed) {
        let tables = tables::tables();
        match self.schedule[index] {
            Task::GateUp { expert, tile } => {
                self.gate_up(tables, expert as usize, tile as usize, speed);
            }
            Task::Down { expert, chunk } => self.down(tables, expert as usize, chunk as usize),
        }
    }

    /// Hidden values `32 tile ..` of `expert` for the rows that use it.
    fn gate_up(&self, tables: &tables::Tables, e: usize, tile: usize, speed: &ThreadSpeed) {
        {
            let gather = match self.features.decode {
                Decode::Fastest => speed.gather[self.kind_index()],
                Decode::Gather => true,
                Decode::Loads => false,
            };
            let expert = &self.experts[e];
            let mut rows = [RowIn {
                q: self.q.as_ptr(),
                ds: self.ds.as_ptr(),
                sum8: self.sum8.as_ptr(),
                planes: self.planes.row(0),
            }; MAX_ROWS];
            let mut hidden = [self.hidden(0); MAX_ROWS];
            for (n, b) in expert.row.iter().enumerate() {
                let b = *b as usize;
                // SAFETY: row b's part of the activations the job copied.
                rows[n] = unsafe {
                    RowIn {
                        q: self.q.as_ptr().add(b * self.n_in),
                        ds: self.ds.as_ptr().add(b * self.n_in / 32 * 8),
                        sum8: self.sum8.as_ptr().wrapping_add(b * self.n_in / 32),
                        planes: self.planes.row(b),
                    }
                };
                hidden[n] = self.hidden(expert.first_pair + n);
            }
            let (n, order) = (expert.row.len(), self.down.order());
            // SAFETY: a written kind is only made where `Features::detect`
            // found x86-64-v3 (`new`); the matrices are the creator's
            // guarantee; the rows and hiddens are the job's own, sized by
            // `new`, and this tile of these pairs is written by this task
            // only.
            unsafe {
                match self.kind {
                    GateKind::Other(ty) => kernels::gate_up_tile_decoded(
                        self.features.vector,
                        ty,
                        expert.gate,
                        expert.up,
                        self.n_in,
                        tile,
                        &rows[..n],
                        &hidden[..n],
                        order,
                    ),
                    kind => kernels::gate_up_tile(
                        tables,
                        kind,
                        gather,
                        self.features.vnni,
                        expert.gate,
                        expert.up,
                        self.n_in,
                        tile,
                        &rows[..n],
                        &hidden[..n],
                        order,
                    ),
                }
            }
            self.tiles_left[e].fetch_sub(1, Ordering::AcqRel);
            if self.gate_up_done.fetch_add(1, Ordering::AcqRel) + 1 == self.gate_up_tasks {
                let ns = self.created.elapsed().as_nanos() as u64;
                self.gate_up_finished.store(ns, Ordering::Relaxed);
            }
        }
    }

    /// Output values `64 chunk ..` of `expert`'s down product, added to the
    /// sums after the parts of the experts before it.
    fn down(&self, tables: &tables::Tables, e: usize, chunk: usize) {
        let expert = &self.experts[e];
        let first = chunk * DOWN_ROWS;
        let count = DOWN_ROWS.min(self.n_out - first);
        let part = DownExpert {
            down: expert.down,
            hidden: &self.hiddens[expert.first_pair..expert.first_pair + expert.row.len()],
            row: &expert.place,
            weight: &expert.weight,
        };
        // The sums are kept by the pairs' places among the expert's pairs.
        let mut sums = [0.0f32; MAX_ROWS * DOWN_ROWS];
        let (n_ff, n_rows, out) = (self.n_ff, expert.place.len(), sums.as_mut_ptr());
        // SAFETY: as for the gate and up tasks; the expert's hidden values are
        // complete (the caller waited for its gate and up tasks), and the
        // sums are this task's own, `DOWN_ROWS` per row.
        unsafe {
            match self.down {
                DownKind::Other(ty) => kernels::down_rows_decoded(
                    self.features.vector,
                    ty,
                    std::slice::from_ref(&part),
                    n_ff,
                    n_rows,
                    first,
                    count,
                    out,
                    DOWN_ROWS,
                ),
                kind => kernels::down_rows(
                    tables,
                    self.features.vnni,
                    kind,
                    std::slice::from_ref(&part),
                    n_ff,
                    n_rows,
                    first,
                    count,
                    out,
                    DOWN_ROWS,
                ),
            }
        }
        // The parts of a chunk are added in the order of the experts: the
        // part before this one is claimed earlier and ends soon.
        while self.chunk_parts[chunk].load(Ordering::Acquire) < e {
            std::hint::spin_loop();
        }
        for (&b, &place) in expert.row.iter().zip(&expert.place) {
            let b = b as usize;
            // SAFETY: values `first .. first + count` of row b are inside the
            // output, and only the part of this chunk whose turn it is
            // writes them.
            let row = unsafe {
                std::slice::from_raw_parts_mut(self.out.ptr.add(b * self.n_out + first), count)
            };
            for (out, part) in row.iter_mut().zip(&sums[usize::from(place) * DOWN_ROWS..]) {
                *out += *part;
            }
        }
        self.chunk_parts[chunk].store(e + 1, Ordering::Release);
    }

    /// Whether every task has been claimed: a thread that finds this has
    /// nothing to do here.
    pub fn is_claimed(&self) -> bool {
        self.next.load(Ordering::Relaxed) >= self.tasks
    }

    /// Claims and runs tasks until none is left. Any number of threads may
    /// call this at once.
    pub fn work(&self) {
        self.work_in(None);
    }

    /// [`LayerJob::work`] as a member of `team`. Tasks are claimed one at a
    /// time, so threads share a job by their speed; but a slow thread that
    /// took one of the last gate and up tiles of the last expert would hold
    /// everyone up for the time of its tile, so it leaves them when the fast
    /// threads will have done what is left sooner than it would finish one.
    pub(crate) fn work_in(&self, team: Option<&Team>) {
        let speed = thread_speed(self.features.vector);
        let kind = self.kind_index();
        let slow = team.is_some_and(|t| is_slow(speed.tile_ns[kind], t.fast_tile_ns[kind]));
        let mut busy = 0u64;
        loop {
            let seen = self.next.load(Ordering::Relaxed);
            if seen >= self.tasks {
                break;
            }
            let last_tiles =
                seen < self.last_gate_up_end && self.last_gate_up_end - seen <= self.n_ff / TILE;
            if let (true, Some(team)) = (slow && last_tiles, team) {
                let left = (self.last_gate_up_end - seen) as u64;
                let by_the_fast =
                    left * u64::from(team.fast_tile_ns[kind]) / u64::from(team.fast[kind].max(1));
                if by_the_fast < u64::from(speed.tile_ns[kind]) {
                    std::hint::spin_loop();
                    continue;
                }
            }
            // A claim past the end is harmless: the counter only grows by
            // one per caller before it stops.
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            if index >= self.tasks {
                break;
            }
            if let Task::Down { expert, .. } = self.schedule[index] {
                // A down product reads every hidden value of its expert: it
                // starts when the expert's last gate and up task has ended.
                while self.tiles_left[expert as usize].load(Ordering::Acquire) > 0 {
                    std::hint::spin_loop();
                }
            }
            let start = Instant::now();
            self.run_task(index, &speed);
            busy += start.elapsed().as_nanos() as u64;
            if self.done.fetch_add(1, Ordering::AcqRel) + 1 == self.tasks {
                let ns = self.created.elapsed().as_nanos() as u64;
                self.finished.store(ns + 1, Ordering::Release);
            }
        }
        if busy > 0 {
            self.busy.fetch_add(busy, Ordering::Relaxed);
        }
    }

    /// Whether every task has run.
    pub fn is_done(&self) -> bool {
        self.finished.load(Ordering::Acquire) != 0
    }

    /// From the creation of the job to the end of its last task; `None`
    /// while it runs.
    pub fn elapsed(&self) -> Option<Duration> {
        match self.finished.load(Ordering::Acquire) {
            0 => None,
            ns => Some(Duration::from_nanos(ns - 1)),
        }
    }

    /// When the job was made.
    pub fn created(&self) -> Instant {
        self.created
    }

    /// When the last task ended; `None` while the job runs.
    pub fn ended(&self) -> Option<Instant> {
        self.elapsed().map(|elapsed| self.created + elapsed)
    }

    /// From the creation of the job to the end of its gate and up products
    /// (the rest of [`LayerJob::elapsed`] is the down products).
    pub fn gate_up_elapsed(&self) -> Duration {
        Duration::from_nanos(self.gate_up_finished.load(Ordering::Relaxed))
    }

    /// The time threads spent computing the job's tasks, summed over the
    /// threads; waiting for a task is not in it.
    pub fn busy(&self) -> Duration {
        Duration::from_nanos(self.busy.load(Ordering::Relaxed))
    }

    /// The result: for every row of the step, `n_embd` values, the sum over
    /// the job's experts the row picked of weight times expert output.
    /// `None` while the job runs.
    pub fn sums(&self) -> Option<&[f32]> {
        // SAFETY: every task has ended, so nobody writes the output any
        // more, and it is the allocation `new` made.
        self.is_done().then(|| unsafe { std::slice::from_raw_parts(self.out.ptr, self.out.len) })
    }
}

/// Bytes of the activations of `n_rows` rows of `n_in` values that a job
/// reads: the codes and the `(d, s)` pairs of [`ares_quant::act`], as
/// `ares_quantize_act` lays them out (the `f32` copy behind them is not read).
pub fn act_bytes(n_in: usize, n_rows: usize) -> usize {
    act::Layout::of((n_rows * n_in) as u64).pairs_end as usize
}

/// Quantizes rows of activations on the host exactly as `ares_quantize_act`
/// does on the device, into the layout a [`LayerJob`] reads: for tests and
/// for measuring the kernels without a device.
pub fn quantize_activations(x: &[f32], n_in: usize) -> Vec<u8> {
    let values = x.len() / n_in * n_in;
    let mut bytes = act::quantize(&x[..values]);
    bytes.truncate(act::Layout::of(values as u64).pairs_end as usize);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::test_rig::*;

    /// A layer of `n_experts` random experts and `n_rows` rows, each row
    /// using a few of the experts: the job's sums against the reference.
    fn check(gate_ty: GgmlType, n_in: usize, n_ff: usize, n_rows: usize, features: Features) {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (n_ff as u64) << 8 ^ n_rows as u64);
        let n_experts = 3;
        let shape = ExpertShape {
            n_embd: n_in as u32,
            n_ff: n_ff as u32,
            n_experts: 16,
            top_k: 2,
            gate_ty,
            down_ty: GgmlType::Q2_0,
        };
        let matrices: Vec<[Vec<u8>; 3]> = (0..n_experts)
            .map(|_| {
                [
                    random_matrix(&mut rng, gate_ty, n_ff * n_in),
                    random_matrix(&mut rng, gate_ty, n_ff * n_in),
                    random_matrix(&mut rng, GgmlType::Q2_0, n_in * n_ff),
                ]
            })
            .collect();
        let x: Vec<f32> = (0..n_rows * n_in).map(|_| rng.activation()).collect();
        let act = quantize_activations(&x, n_in);
        let xq = dequantized_act(&act, n_in, n_rows);

        // Row b uses expert e unless (b + e) is a multiple of 3; every
        // expert keeps at least row 0 or 1.
        let mut experts = Vec::new();
        for (e, m) in matrices.iter().enumerate() {
            let mut rows = 0u32;
            let mut weight = [0.0f32; MAX_ROWS];
            for (b, weight) in weight.iter_mut().enumerate().take(n_rows) {
                if !(b + e).is_multiple_of(3) || n_rows == 1 {
                    rows |= 1 << b;
                    *weight = 0.2 + rng.unit();
                }
            }
            experts.push(CpuExpert {
                gate: m[0].as_ptr(),
                up: m[1].as_ptr(),
                down: m[2].as_ptr(),
                rows,
                weight,
            });
        }
        // SAFETY: the matrices are whole and outlive the job.
        let job =
            unsafe { LayerJob::new(features, &shape, n_rows, &act, &experts) }.expect("a job");
        job.work();
        let sums = job.sums().expect("the job has run");

        let mut worst_same = 0.0f64;
        let mut worst_exact = 0.0f64;
        for b in 0..n_rows {
            let mut same = vec![0.0f64; n_in];
            let mut exact = vec![0.0f64; n_in];
            let unrounded: Vec<f64> =
                x[b * n_in..(b + 1) * n_in].iter().map(|v| f64::from(*v)).collect();
            for (expert, m) in experts.iter().zip(&matrices) {
                if expert.rows >> b & 1 == 0 {
                    continue;
                }
                let (g, u, d) = (
                    decoded(gate_ty, &m[0]),
                    decoded(gate_ty, &m[1]),
                    decoded(GgmlType::Q2_0, &m[2]),
                );
                let w = f64::from(expert.weight[b]);
                let part = reference(&g, &u, &d, &xq[b * n_in..(b + 1) * n_in], n_ff, true);
                same.iter_mut().zip(&part).for_each(|(s, p)| *s += w * p);
                let part = reference(&g, &u, &d, &unrounded, n_ff, false);
                exact.iter_mut().zip(&part).for_each(|(s, p)| *s += w * p);
            }
            let got: Vec<f64> =
                sums[b * n_in..(b + 1) * n_in].iter().map(|v| f64::from(*v)).collect();
            worst_same = worst_same.max(relative_rms(&got, &same));
            worst_exact = worst_exact.max(relative_rms(&got, &exact));
        }
        // Against the same rounding of activations and hidden values only
        // the order of float additions differs (and, rarely, a hidden code
        // that sat on a rounding boundary).
        assert!(
            worst_same < 2e-3,
            "{gate_ty} {n_rows} rows: {worst_same:e} against the same rounding"
        );
        // Against arithmetic that rounds nothing: the two int8 stages.
        assert!(
            worst_exact < 0.05,
            "{gate_ty} {n_rows} rows: {worst_exact:e} against exact arithmetic"
        );
    }

    #[test]
    fn experts_agree_with_the_reference() {
        let features = Features::detect();
        for ty in [GgmlType::IQ2_S, GgmlType::IQ2_XXS, GgmlType::IQ1_M] {
            for rows in [1, 2, 5, 8] {
                // 192 is an odd number of Q2_0 blocks.
                check(ty, 512, 192, rows, features);
                check(ty, 256, 128, rows, features.without_vnni().with_decode(Decode::Loads));
                check(ty, 256, 128, rows, features.with_decode(Decode::Gather));
            }
        }
    }

    #[test]
    fn a_pool_computes_what_one_thread_does() {
        let mut rng = Rng(77);
        let (n_in, n_ff) = (512, 256);
        let shape = ExpertShape {
            n_embd: n_in as u32,
            n_ff: n_ff as u32,
            n_experts: 16,
            top_k: 2,
            gate_ty: GgmlType::IQ2_S,
            down_ty: GgmlType::Q2_0,
        };
        let m = [
            random_matrix(&mut rng, GgmlType::IQ2_S, n_ff * n_in),
            random_matrix(&mut rng, GgmlType::IQ2_S, n_ff * n_in),
            random_matrix(&mut rng, GgmlType::Q2_0, n_in * n_ff),
        ];
        let x: Vec<f32> = (0..2 * n_in).map(|_| rng.activation()).collect();
        let act = quantize_activations(&x, n_in);
        let expert = CpuExpert {
            gate: m[0].as_ptr(),
            up: m[1].as_ptr(),
            down: m[2].as_ptr(),
            rows: 0b11,
            weight: [0.5; MAX_ROWS],
        };
        let pool = CpuPool::new(&Placement::unpinned(3), Duration::from_millis(1)).expect("a pool");
        // SAFETY: the matrices are whole and outlive both jobs.
        let alone =
            unsafe { LayerJob::new(pool.features(), &shape, 2, &act, &[expert]) }.expect("a job");
        alone.work();
        for _ in 0..20 {
            // SAFETY: as above.
            let job = std::sync::Arc::new(
                unsafe { LayerJob::new(pool.features(), &shape, 2, &act, &[expert]) }
                    .expect("a job"),
            );
            pool.submit(std::sync::Arc::clone(&job));
            let start = Instant::now();
            while !job.is_done() {
                assert!(
                    start.elapsed() < Duration::from_secs(20),
                    "the pool never finished the job"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(job.sums(), alone.sums());
        }
    }

    /// A job of many rows gives every row what jobs of at most eight rows give it: the
    /// expert of eleven rows in two pieces, the others one each, rows in no order.
    #[test]
    fn a_job_of_many_rows_gives_what_jobs_of_eight_give() {
        let mut rng = Rng(91);
        let (n_in, n_ff, n_rows) = (512, 256, 20);
        let shape = ExpertShape {
            n_embd: n_in as u32,
            n_ff: n_ff as u32,
            n_experts: 16,
            top_k: 2,
            gate_ty: GgmlType::IQ2_S,
            down_ty: GgmlType::Q2_0,
        };
        let experts: Vec<[Vec<u8>; 3]> = (0..3)
            .map(|_| {
                [
                    random_matrix(&mut rng, GgmlType::IQ2_S, n_ff * n_in),
                    random_matrix(&mut rng, GgmlType::IQ2_S, n_ff * n_in),
                    random_matrix(&mut rng, GgmlType::Q2_0, n_in * n_ff),
                ]
            })
            .collect();
        let x: Vec<f32> = (0..n_rows * n_in).map(|_| rng.activation()).collect();
        let act = quantize_activations(&x, n_in);
        let rows: [Vec<u32>; 3] =
            [(0..11).map(|r| (r * 7 + 3) % 20).collect(), vec![19, 2], vec![5]];
        let weight = |e: usize, row: u32| 0.25 + 0.05 * e as f32 + 0.01 * row as f32;
        let features = Features::detect();
        let mut pieces = Vec::new();
        for (e, m) in experts.iter().enumerate() {
            for piece in rows[e].chunks(MAX_ROWS) {
                pieces.push(RowsExpert {
                    gate: m[0].as_ptr(),
                    up: m[1].as_ptr(),
                    down: m[2].as_ptr(),
                    rows: piece.iter().map(|r| (*r, weight(e, *r))).collect(),
                });
            }
        }
        // SAFETY: the matrices are whole and outlive the jobs.
        let wide =
            unsafe { LayerJob::with_rows(features, &shape, n_rows, &act, &pieces) }.expect("a job");
        wide.work();
        let wide = wide.sums().expect("done");
        // Each row by itself, in a job of one row.
        for row in 0..n_rows as u32 {
            let x_row = &x[row as usize * n_in..(row as usize + 1) * n_in];
            let act_row = quantize_activations(x_row, n_in);
            let one: Vec<CpuExpert> = experts
                .iter()
                .enumerate()
                .filter(|(e, _)| rows[*e].contains(&row))
                .map(|(e, m)| CpuExpert {
                    gate: m[0].as_ptr(),
                    up: m[1].as_ptr(),
                    down: m[2].as_ptr(),
                    rows: 1,
                    weight: [weight(e, row); MAX_ROWS],
                })
                .collect();
            let got = &wide[row as usize * n_in..(row as usize + 1) * n_in];
            if one.is_empty() {
                assert!(got.iter().all(|v| *v == 0.0), "row {row} picked nothing");
                continue;
            }
            // SAFETY: as above.
            let job = unsafe { LayerJob::new(features, &shape, 1, &act_row, &one) }.expect("a job");
            job.work();
            let want = job.sums().expect("done");
            for (j, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "row {row} value {j}: {g} {w}");
            }
        }
    }
}
