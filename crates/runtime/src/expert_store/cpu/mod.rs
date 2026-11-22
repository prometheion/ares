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
//!   IQ2_S, IQ2_XXS, IQ1_M, IQ3_XXS, IQ4_XS, Q4_K and Q5_K gate and up and for Q2_0,
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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_cuda::ExpertShape;

use kernels::{DownExpert, DownKind, GateKind, Hidden, Order, RowIn};
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
/// for them (IQ2_S, IQ2_XXS, IQ1_M, IQ3_XXS, IQ4_XS, Q4_K, Q5_K).
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
    let row = RowIn { q: q.as_ptr(), ds: act[ds_at..].as_ptr(), sum8: sum8.as_ptr() };
    let (mut hq, mut hsum, mut hscale) = ([0i8; 128], [0i32; 8], [0.0f32; 8]);
    let hidden = Hidden { q: hq.as_mut_ptr(), sum: hsum.as_mut_ptr(), scale: hscale.as_mut_ptr() };
    let mut speed = ThreadSpeed { gather: [true; MEASURED], tile_ns: [u32::MAX; MEASURED] };
    for (k, kind) in GateKind::WRITTEN.into_iter().enumerate() {
        let bytes = 32 * (n_in / 256) * kind.block_bytes();
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
    /// The pairs of the expert are `first_pair ..`, one per row in `row`.
    first_pair: usize,
    row: Vec<u8>,
    weight: Vec<f32>,
}

/// The experts of one layer that the CPU computes for one step.
///
/// The job owns a copy of the rows' activations and everything it computes;
/// the weights it only points at. Its work is a list of tasks (first the
/// gate and up product of every expert in pieces of 32 hidden values, then
/// the down products in pieces of 64 output values) that threads claim one
/// by one, so however many take part, each task runs once and the result
/// does not depend on who ran it.
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
    hidden_q: Scratch<i8>,
    hidden_sum: Scratch<i32>,
    hidden_scale: Scratch<f32>,
    /// Hidden values per pair in the arrays above: `n_ff` rounded up to 128.
    hidden_stride: usize,
    out: Scratch<f32>,
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
        let (n_in, n_ff) = (shape.n_embd as usize, shape.n_ff as usize);
        if n_rows == 0 || n_rows > MAX_ROWS {
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

        let mut list = Vec::with_capacity(experts.len());
        let mut pairs = 0;
        for expert in experts {
            let rows: Vec<u8> = (0..n_rows as u8).filter(|b| expert.rows >> b & 1 != 0).collect();
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
                first_pair: pairs,
                weight: rows.iter().map(|b| expert.weight[usize::from(*b)]).collect(),
                row: rows,
            });
            pairs += list.last().map_or(0, |e| e.row.len());
        }
        let hidden_stride = n_ff.next_multiple_of(128);
        let gate_up_tasks = list.len() * (n_ff / TILE);
        let tasks = gate_up_tasks + if list.is_empty() { 0 } else { n_in.div_ceil(DOWN_ROWS) };
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
            hidden_q: Scratch::new(pairs * hidden_stride, 0),
            hidden_sum: Scratch::new(pairs * hidden_stride / 16, 0),
            hidden_scale: Scratch::new(pairs * hidden_stride / 16, 0.0),
            hidden_stride,
            out: Scratch::new(n_rows * n_in, 0.0),
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
        let at = pair * self.hidden_stride;
        // SAFETY: a pair's part of the three arrays, inside them by how
        // `new` sized them.
        unsafe {
            Hidden {
                q: self.hidden_q.ptr.add(at),
                sum: self.hidden_sum.ptr.add(at / 16),
                scale: self.hidden_scale.ptr.add(at / 16),
            }
        }
    }

    /// The index of the job's gate type in a [`ThreadSpeed`].
    fn kind_index(&self) -> usize {
        self.kind.index()
    }

    /// Runs task `index`, which the caller has claimed. A down task must
    /// not start before the last gate and up task has ended.
    fn run_task(&self, index: usize, speed: &ThreadSpeed) {
        let tables = tables::tables();
        if index < self.gate_up_tasks {
            let gather = match self.features.decode {
                Decode::Fastest => speed.gather[self.kind_index()],
                Decode::Gather => true,
                Decode::Loads => false,
            };
            let tiles = self.n_ff / TILE;
            let expert = &self.experts[index / tiles];
            let mut rows =
                [RowIn { q: self.q.as_ptr(), ds: self.ds.as_ptr(), sum8: self.sum8.as_ptr() };
                    MAX_ROWS];
            let mut hidden = [self.hidden(0); MAX_ROWS];
            for (n, b) in expert.row.iter().enumerate() {
                let b = usize::from(*b);
                // SAFETY: row b's part of the activations the job copied.
                rows[n] = unsafe {
                    RowIn {
                        q: self.q.as_ptr().add(b * self.n_in),
                        ds: self.ds.as_ptr().add(b * self.n_in / 32 * 8),
                        sum8: self.sum8.as_ptr().wrapping_add(b * self.n_in / 32),
                    }
                };
                hidden[n] = self.hidden(expert.first_pair + n);
            }
            let (n, tile, order) = (expert.row.len(), index % tiles, self.down.order());
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
            if self.gate_up_done.fetch_add(1, Ordering::AcqRel) + 1 == self.gate_up_tasks {
                let ns = self.created.elapsed().as_nanos() as u64;
                self.gate_up_finished.store(ns, Ordering::Relaxed);
            }
        } else {
            let first = (index - self.gate_up_tasks) * DOWN_ROWS;
            let count = DOWN_ROWS.min(self.n_out - first);
            let hiddens: Vec<Hidden> = (0..self.pairs()).map(|p| self.hidden(p)).collect();
            let experts: Vec<DownExpert<'_>> = self
                .experts
                .iter()
                .map(|e| DownExpert {
                    down: e.down,
                    hidden: &hiddens[e.first_pair..e.first_pair + e.row.len()],
                    row: &e.row,
                    weight: &e.weight,
                })
                .collect();
            let (n_ff, n_out, n_rows, out) = (self.n_ff, self.n_out, self.n_rows, self.out.ptr);
            // SAFETY: as above; the hidden values are complete, and this
            // range of the output is this task's alone.
            unsafe {
                match self.down {
                    DownKind::Other(ty) => kernels::down_rows_decoded(
                        self.features.vector,
                        ty,
                        &experts,
                        n_ff,
                        n_out,
                        n_rows,
                        first,
                        count,
                        out,
                    ),
                    kind => kernels::down_rows(
                        tables,
                        self.features.vnni,
                        kind,
                        &experts,
                        n_ff,
                        n_out,
                        n_rows,
                        first,
                        count,
                        out,
                    ),
                }
            }
        }
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
    /// took one of the last gate and up tiles would hold everyone up for the
    /// time of its tile, so it leaves them when the fast threads will have
    /// done what is left sooner than it would finish one.
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
            if let (true, Some(team)) = (slow && seen < self.gate_up_tasks, team) {
                let left = (self.gate_up_tasks - seen) as u64;
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
            if index >= self.gate_up_tasks {
                // The down products read every hidden value: they start
                // when the last gate and up task has ended.
                while self.gate_up_done.load(Ordering::Acquire) < self.gate_up_tasks {
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
/// reads: the codes and the `(d, s)` pairs, as `ares_quantize_act` lays them
/// out (the `f32` copy behind them is not read).
pub fn act_bytes(n_in: usize, n_rows: usize) -> usize {
    let codes = n_rows * n_in;
    codes.next_multiple_of(16) + codes / 32 * 8
}

/// Quantizes rows of activations on the host exactly as `ares_quantize_act`
/// does on the device, into the layout a [`LayerJob`] reads: for tests and
/// for measuring the kernels without a device.
pub fn quantize_activations(x: &[f32], n_in: usize) -> Vec<u8> {
    let n_rows = x.len() / n_in;
    let mut act = vec![0u8; act_bytes(n_in, n_rows)];
    let ds_at = (n_rows * n_in).next_multiple_of(16);
    for (chunk, values) in x.as_chunks::<32>().0.iter().enumerate() {
        let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let mut sum = 0.0f32;
        for (i, v) in values.iter().enumerate() {
            let code = if amax == 0.0 { 0.0 } else { (v / d).round() };
            act[32 * chunk + i] = code as i8 as u8;
            sum += code;
        }
        act[ds_at + 8 * chunk..ds_at + 8 * chunk + 4].copy_from_slice(&d.to_le_bytes());
        act[ds_at + 8 * chunk + 4..ds_at + 8 * chunk + 8].copy_from_slice(&(d * sum).to_le_bytes());
    }
    act
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small generator with a fixed sequence (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        /// Roughly normal, with a rare value ten times as large.
        fn activation(&mut self) -> f32 {
            let v: f32 = (0..6).map(|_| self.unit() - 0.5).sum();
            if self.next().is_multiple_of(61) { 10.0 * v } else { v }
        }
    }

    /// Random blocks of `ty` with scales of moderate size.
    fn random_matrix(rng: &mut Rng, ty: GgmlType, values: usize) -> Vec<u8> {
        let block = ty.block_bytes() as usize;
        let mut bytes = vec![0u8; values / ty.block_values() as usize * block];
        for b in &mut bytes {
            *b = rng.next() as u8;
        }
        for blk in bytes.chunks_exact_mut(block) {
            // An f16 between 1/64 and 1/8, wherever the type keeps it.
            let d = (0x2400u16 + (rng.next() % 0xC00) as u16).to_le_bytes();
            match ty {
                GgmlType::IQ1_M => {
                    blk[49] = blk[49] & 0x0f | (d[0] & 0x0f) << 4;
                    blk[51] = blk[51] & 0x0f | d[0] & 0xf0;
                    blk[53] = blk[53] & 0x0f | (d[1] & 0x0f) << 4;
                    blk[55] = blk[55] & 0x0f | d[1] & 0xf0;
                }
                _ => blk[..2].copy_from_slice(&d),
            }
        }
        bytes
    }

    fn decoded(ty: GgmlType, bytes: &[u8]) -> Vec<f32> {
        let mut out =
            vec![0.0; bytes.len() / ty.block_bytes() as usize * ty.block_values() as usize];
        ares_quant::dequantize(ty, bytes, &mut out).expect("whole blocks");
        out
    }

    /// The activations the kernels see: codes times scales.
    fn dequantized_act(act: &[u8], n_in: usize, n_rows: usize) -> Vec<f64> {
        let ds_at = (n_rows * n_in).next_multiple_of(16);
        (0..n_rows * n_in)
            .map(|i| {
                let at = ds_at + 8 * (i / 32);
                let d = f32::from_le_bytes([act[at], act[at + 1], act[at + 2], act[at + 3]]);
                f64::from(act[i] as i8) * f64::from(d)
            })
            .collect()
    }

    /// The expert in exact arithmetic on the decoded weights. `quantized`
    /// rounds the hidden values as the kernels do; without it nothing is
    /// rounded.
    fn reference(
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        x: &[f64],
        n_ff: usize,
        quantized: bool,
    ) -> Vec<f64> {
        let n_in = x.len();
        let dot =
            |w: &[f32], x: &[f64]| w.iter().zip(x).map(|(w, x)| f64::from(*w) * x).sum::<f64>();
        let mut h: Vec<f64> = (0..n_ff)
            .map(|j| {
                let g = dot(&gate[j * n_in..(j + 1) * n_in], x);
                let u = dot(&up[j * n_in..(j + 1) * n_in], x);
                if quantized {
                    let (g, u) = (g as f32, u as f32);
                    f64::from(g / (1.0 + (-g).exp()) * u)
                } else {
                    g / (1.0 + (-g).exp()) * u
                }
            })
            .collect();
        if quantized {
            for chunk in h.as_chunks_mut::<32>().0.iter_mut() {
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs() as f32));
                let d = amax / 127.0;
                for v in chunk {
                    *v = if amax > 0.0 { f64::from((*v as f32 / d).round() * d) } else { 0.0 };
                }
            }
        }
        (0..n_in).map(|i| dot(&down[i * n_ff..(i + 1) * n_ff], &h)).collect()
    }

    fn relative_rms(got: &[f64], want: &[f64]) -> f64 {
        let err: f64 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
        let norm: f64 = want.iter().map(|w| w * w).sum();
        (err / norm).sqrt()
    }

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
}
