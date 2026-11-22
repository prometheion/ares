//! Which layout each weight matrix gets on a device: decided per matrix, by
//! timing both layouts on the device when the model is loaded.
//!
//! A matrix whose type has a device layout (`ares_cuda::WeightLayout::Device`)
//! can lie on the device as the file stores it or rearranged for the tensor
//! cores. The kernels over the device layout serve every number of rows from
//! one copy, at the exact arithmetic of f16 operands, are far ahead from a few
//! rows on, and with one row are within a few percent of the integer decode
//! kernels of the file's layout either way: slightly behind on a few shapes of
//! a card, ahead on others. Which it is depends on the type, the shape and the
//! card, so it is measured, before the weights are placed:
//!
//! - once per type, which path of the tile kernels multiplies it in the file's
//!   layout (`Kernels::measure_prefill_path`, 30 to 80 ms), and which kernel
//!   of the device layout serves which number of rows above those of a
//!   decode step (`Kernels::measure_layout`, about 50 ms),
//!   and once per type and shape how the kernels for a decode step launch
//!   (`Kernels::tune_layout`, a few milliseconds);
//! - once per type and shape, the product in both layouts with the rows the
//!   model will run: those of a decode step and those of a group of prompt
//!   rows. The probe is a matrix of that shape (cut to [`PROBE_WEIGHT_BYTES`]
//!   and [`PROBE_RESULT_BYTES`], above which the time is what the memory
//!   gives), filled with constants, and the products are the very calls of a
//!   step;
//! - once per width of an input, what preparing it costs in each form: a
//!   launch of its own, or the fused kernel that writes the one or the other.
//!
//! **Where the probe reads.** A step reads every matrix once, each from its
//! own place in gigabytes of weights. A probe that reads one place again and
//! again finds it in the L2 cache, and one that walks a few dozen megabytes
//! still finds the translations of its addresses: both flatter the device
//! layout, whose tiles read codes and headers from two places. (Measured on
//! an RTX 4070: a Q4_K matrix of 4096 x 1024 with one row, file's layout
//! against device layout: 5.4 against 4.4 microseconds walking 72 MiB, 7.2
//! against 7.3 walking 1 GiB or 8; and Qwen3.8-27B decoded 1.7 ms per step
//! slower in the device layout than probes on 72 MiB had it.) So the probe's
//! weights are a buffer as large as the device's weights will be, and every
//! launch reads the place behind the one before, as the matrices of a step
//! follow each other.
//!
//! **Matrices that read the same input** are decided together, because they
//! share its preparation: the file's layout reads 8-bit activations at a few
//! rows and the device layout f16, and a group that mixes the two prepares
//! both. Under a tensor split the two groups of a half block (the matrices
//! that read the norm, and the one that reads what they lead to) are decided
//! together too, because the devices meet after every half. The assignments
//! that are costed: every matrix in the file's layout, every matrix in the
//! device layout, and each in the one that is cheaper for itself, per group.
//!
//! **The weighting.** An assignment has two times, that of its products and
//! preparations in a decode step and that in a group of prompt rows, and
//! each is taken relative to what it is with every matrix in the file's
//! layout. Decode steps come first: a token is waited for one step at a
//! time, so by default an assignment is taken only if it does not make the
//! decode step slower, and of those the one that reads prompts fastest. What
//! "not slower" means is [`AS_GOOD_WITHIN`], two percent of the group's
//! time; of assignments that are equal in both, the one with the most
//! matrices in the device layout is taken, because there the exact
//! arithmetic costs nothing. The decode rows are one, or the rows of a
//! window when the model verifies its own drafts in every step, where the
//! device layout is rarely the slower one.
//!
//! [`Layouts::Measured`] carries the factor by which prompts may buy decode
//! time, for whoever serves long prompts: with a factor `w` the cost of an
//! assignment is its decode fraction plus `w` times its prompt fraction, so
//! `w = 1` takes an assignment that makes decode steps 3 % slower and prompts
//! 10 % faster and refuses one that makes them 3 % slower and 2 % faster. No
//! constant of a machine enters either way.
//!
//! **Prompts wait for the slowest device.** A prompt is read by the devices
//! of a tensor split no faster than by the slowest of them, so only that
//! device's gain is a gain. The devices therefore tell each other, while
//! they load, what a group of prompt rows costs them with every matrix in
//! the file's layout ([`Balance`]), and each takes its prompt fraction
//! against the larger of its own time and the slowest other device's. (The
//! same holds for the halves of a decode step, where the devices meet twice
//! per block, but what a half costs besides its weight products depends on
//! the length of the context, so no device counts on another's lead there.)
//!
//! **Exact products** ([`Precision::Exact`]) are the engine's reference and
//! are not chosen by a clock: every matrix that has a device layout takes it,
//! with the kernels of the table that is kept, so that two loads of a model
//! compute the same bits.
//!
//! The probe takes as much of the device as the weights will, and about
//! 150 MiB for activations and results, and frees all of it before the
//! weights are placed.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_cuda::{
    ActivationForm, Device, DeviceBuffer, DevicePtr, Event, Kernels, LayerKernels, MAX_BATCH,
    OutputGate, Precision, PrefillChoice, PrefillPath, Stream, WeightLayout,
};
use ares_runtime::ModelError;

/// How the layout of each weight matrix is chosen when a model is loaded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layouts {
    /// Per matrix, by timing both layouts on the device (see the module).
    Measured {
        /// What the speed of prompts weighs against the speed of decode
        /// steps. 0, the default, never makes a decode step slower and takes
        /// what prompts gain within that; 1 counts a percent of either the
        /// same; more favours prompts further.
        prompt_weight: f32,
    },
    /// Every matrix as the file stores it. For comparisons.
    File,
    /// The device layout for every matrix that can have it. For comparisons.
    Device,
}

impl Default for Layouts {
    fn default() -> Self {
        Self::Measured { prompt_weight: 0.0 }
    }
}

/// An assignment whose cost (a fraction that is 1 in the file's layout, or
/// the weighted sum of two) is within this of the cheapest counts as as
/// good; of those the one that reads prompts fastest is taken, and of equals
/// the one with the most matrices in the device layout. It is what the
/// choice may cost a group's decode time at most, and the noise of a probe:
/// the same shape measured in two loads differs by one to three percent
/// (clocks, and whatever else ran), and at half a percent a third of the
/// matrices were left in the file's layout by that noise, a different third
/// in every load, which made the numbers of two loads of a model differ.
const AS_GOOD_WITHIN: f64 = 0.02;
/// The most bytes of weights (or the L2 cache and a quarter, if that is more)
/// and of results a probe product moves. A larger matrix is probed with fewer
/// rows; both layouts are bound by memory there.
const PROBE_WEIGHT_BYTES: u64 = 64 << 20;
const PROBE_RESULT_BYTES: u64 = 32 << 20;
/// What the probe leaves of the device's free memory for its other buffers
/// and for whatever else is allocated while a model loads.
const PROBE_HEADROOM_BYTES: u64 = 768 << 20;
/// The most prompt rows of a probe: from 256 rows on the tensor-core kernels
/// of both layouts run at their rate.
const PROBE_PROMPT_ROWS: u32 = 256;
/// A product is timed this many times as a run of launches; its fastest run
/// counts.
const ROUNDS: usize = 3;
const BURST_FEW: u32 = 8;
const BURST_MANY: u32 = 2;
/// How long a device waits for the others' times before it decides alone.
const PATIENCE: Duration = Duration::from_secs(120);

/// How a matrix in the file's layout is multiplied with `n` rows: `None` for
/// the decode kernels, or the path of the tile kernels.
pub(crate) fn file_path(
    kernels: &Kernels,
    precision: Precision,
    ty: GgmlType,
    n: u32,
) -> Result<Option<PrefillPath>, ModelError> {
    match precision {
        Precision::Exact => Ok(Some(kernels.prefill_path(Precision::Exact)?)),
        Precision::Fast => {
            let decode = Kernels::decode_kernel_rows(ty)
                .or_else(|| Kernels::has_fast_matvec(ty).then_some(MAX_BATCH));
            match decode {
                Some(limit) if n <= limit.min(MAX_BATCH) => Ok(None),
                _ => Ok(Some(kernels.prefill_path_for(ty, Precision::Fast)?)),
            }
        }
    }
}

/// Whether matrices of `types` in the file's layout take `n` rows through
/// the decode kernels in several launches of up to [`MAX_BATCH`] rows. That
/// is so for a few more rows than one launch takes, where every type's
/// decode kernel still beats its tile kernel (`Kernels::decode_kernel_rows`);
/// from there on the tile kernels multiply the file's layout, whatever the
/// type. (`Precision::Exact` multiplies with the tile kernels' f16 path and
/// has no such window.)
pub(crate) fn decode_in_batches(
    precision: Precision,
    mut types: impl Iterator<Item = GgmlType>,
    n: u32,
) -> bool {
    matches!(precision, Precision::Fast)
        && n > MAX_BATCH
        && types.all(|ty| Kernels::decode_kernel_rows(ty).is_some_and(|limit| n <= limit))
}

/// What a matrix is used for, as far as its layout depends on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Usage {
    /// The rows of the products of a decode step.
    pub decode_rows: u32,
    /// Whether the matrix multiplies the rows of prompts.
    pub prompt: bool,
}

/// How the input of a group of matrices gets its prepared forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Prep {
    /// The merge of a tensor split writes the forms the group reads, in the
    /// launch that sums the partial results: a form costs nothing.
    Merge,
    /// The gated norm of a recurrent block with this many heads of this
    /// width writes the one form its one reader takes: one of two kernels.
    GatedNorm { heads: u32, width: u32 },
    /// SwiGLU writes the one form its one reader takes: one of two kernels.
    Swiglu,
    /// A launch per form (`Kernels::prepare_activations`).
    Launch,
}

/// The type, rows and columns of a matrix.
pub(crate) type Shape = (GgmlType, u32, u32);

/// One probed shape and what was measured for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutProbe {
    pub ty: GgmlType,
    pub n_out: u32,
    pub n_in: u32,
    /// The rows of a decode step and of a group of prompt rows (0: the matrix
    /// reads no prompts) the products were timed with.
    pub decode_rows: u32,
    pub prompt_rows: u32,
    /// Milliseconds of the decode product and of the prompt product in the
    /// file's layout; infinite where it has no kernel for the rows.
    pub file_ms: [f32; 2],
    /// The same in the device layout; for a type without one, the file's.
    pub device_ms: [f32; 2],
    /// The matrices of this shape on the device that got the file's layout
    /// and the device layout.
    pub matrices: [u32; 2],
}

/// What the choice of layouts did on one device.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayoutReport {
    /// The shapes that were probed.
    pub probes: Vec<LayoutProbe>,
    /// What preparing an input by a launch of its own was measured to take:
    /// the width of the input, its rows, and the milliseconds for 8-bit
    /// activations and for f16.
    pub preparations: Vec<(u32, u32, [f32; 2])>,
    /// The matrices in each layout and their bytes on the device:
    /// `[file, device]`. The rows of the embedding table that the draft chain
    /// keeps count as the file's (a tied head's rows, which they are, count as
    /// the head), and a head in two parts is two matrices.
    pub matrices: [u32; 2],
    pub bytes: [u64; 2],
    /// The bytes the device layout added: rows rounded up to 16.
    pub padding: u64,
    /// The path of the tile kernels each type takes in the file's layout
    /// (`Kernels::measure_prefill_path`).
    pub paths: Vec<PrefillChoice>,
    /// How long the tables of the kernels and the probes took.
    pub tables: Duration,
    pub probes_time: Duration,
    /// The bytes of weights the probe walked.
    pub probe_bytes: u64,
    /// What the probes add up to for the weight products of a decode step
    /// and of a group of prompt rows on this device, in milliseconds: with
    /// every matrix in the file's layout, and as chosen.
    pub decode_ms: [f32; 2],
    pub prompt_ms: [f32; 2],
    /// The other devices whose times this device knew when it chose.
    pub balanced_with: usize,
}

/// What a device tells the others of a tensor split: the milliseconds of its
/// weight products for a group of prompt rows, with every matrix in the
/// file's layout.
#[derive(Clone, Debug, Default, PartialEq)]
struct Told {
    prompt: f32,
    /// Whether the device has a time to tell (it measures its layouts).
    known: bool,
}

/// Where the devices of a tensor split tell each other, while each loads its
/// weights, what the weight products cost it: see the module. One per model;
/// every device tells once per load and waits for the others.
#[derive(Debug, Default)]
pub struct Balance {
    state: Mutex<BalanceState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct BalanceState {
    told: Vec<Option<Told>>,
    /// The devices that have read the round that is complete.
    read: usize,
}

impl Balance {
    /// For `devices` devices.
    pub fn new(devices: usize) -> Self {
        Self {
            state: Mutex::new(BalanceState { told: vec![None; devices], read: 0 }),
            changed: Condvar::new(),
        }
    }

    /// Tells the others that device `index` will tell nothing, because it
    /// failed before it could: they decide on their own at once, where they
    /// would wait for it for two minutes. Does not wait.
    pub fn give_up(&self, index: usize) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = state.told.get_mut(index)
            && slot.is_none()
        {
            *slot = Some(Told::default());
            self.changed.notify_all();
        }
    }

    /// Tells the others what device `index` measured and returns what every
    /// device told, once all have; `None` when one does not tell in time.
    fn exchange(&self, index: usize, told: Told) -> Option<Vec<Told>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if index >= state.told.len() {
            return None;
        }
        state.told[index] = Some(told);
        self.changed.notify_all();
        let begun = Instant::now();
        while state.told.iter().any(Option::is_none) {
            let left = PATIENCE.checked_sub(begun.elapsed())?;
            state = self.changed.wait_timeout(state, left).unwrap_or_else(|e| e.into_inner()).0;
        }
        let all: Vec<Told> = state.told.iter().flatten().cloned().collect();
        // When every device has read the round, the next load starts anew.
        state.read += 1;
        if state.read == state.told.len() {
            state.told.fill(None);
            state.read = 0;
        }
        Some(all)
    }
}

/// A buffer of the probe that grows to what a shape needs.
#[derive(Default)]
struct Grown(Option<DeviceBuffer>);

impl Grown {
    fn ensure(&mut self, device: &Device, bytes: u64) -> Result<DevicePtr, ModelError> {
        if self.0.as_ref().is_none_or(|b| b.len() < bytes) {
            // Free the old one first: the probe must not hold both.
            self.0 = None;
            self.0 = Some(DeviceBuffer::new(device, bytes.max(64))?);
        }
        Ok(self.0.as_ref().expect("allocated above").ptr())
    }

    fn len(&self) -> u64 {
        self.0.as_ref().map_or(0, DeviceBuffer::len)
    }
}

/// The device memory and the clock of the probes.
struct Probe {
    stream: Stream,
    start: Event,
    end: Event,
    /// The weights: as many bytes as the device's weights will be.
    w: DeviceBuffer,
    /// Where in the weights the next launch reads.
    cursor: u64,
    x: Grown,
    act_q: Grown,
    act_h: Grown,
    y: Grown,
}

impl Probe {
    /// The place of a matrix of `bytes` bytes for the next launch: behind the
    /// one before, from the start again at the end of the buffer.
    fn place(&mut self, bytes: u64) -> DevicePtr {
        let bytes = bytes.next_multiple_of(64);
        if self.cursor + bytes > self.w.len() {
            self.cursor = 0;
        }
        let at = self.w.ptr().offset(self.cursor);
        self.cursor += bytes;
        at
    }

    /// The milliseconds of one of `burst` runs of `launches`.
    fn run(
        &mut self,
        burst: u32,
        mut launches: impl FnMut(&mut Self) -> Result<(), ModelError>,
    ) -> Result<f32, ModelError> {
        self.start.record(&self.stream)?;
        for _ in 0..burst {
            launches(self)?;
        }
        self.end.record(&self.stream)?;
        self.end.synchronize()?;
        Ok(self.end.elapsed_ms_since(&self.start)? / burst as f32)
    }
}

/// What was measured for one shape: where its probe is in the report,
/// whether the matrix can have a device layout, and which forms the file's
/// layout reads at the decode rows and at the prompt rows (`true`: f16).
#[derive(Clone, Copy, Debug)]
struct Timed {
    probe: usize,
    eligible: bool,
    file_f16: [bool; 2],
}

/// The matrices that read one input, as the loader asked for their layouts.
#[derive(Clone, Debug)]
struct Group {
    shapes: Vec<Shape>,
    usage: Usage,
    prep: Prep,
    /// The half block of the model the group belongs to, under a tensor
    /// split; `None` for a group that is decided on its own.
    half: Option<u32>,
}

/// A group as the decision sees it: its probes, the cost of preparing its
/// input in each form at the decode rows and at the prompt rows, and the
/// assignments to choose from; the last is the file's layout.
struct Costed {
    timed: Vec<Timed>,
    rows: [u32; 2],
    prep_ms: [[f32; 2]; 2],
    /// Whether a form costs its preparation whenever a matrix reads it
    /// (otherwise the one kernel that writes either costs one of the two).
    per_form: bool,
    candidates: Vec<Vec<WeightLayout>>,
}

/// Chooses the layouts of the matrices of one device.
///
/// The loader lays its tensors out twice. The first time the chooser only
/// notes the groups it is asked about ([`Chooser::group`] answers with the
/// file's layout); [`Chooser::decide`] then probes and decides, and the
/// second time the answers are the decisions, in the same order.
pub(crate) struct Chooser<'a> {
    device: &'a Device,
    kernels: &'a Kernels,
    layers: &'a LayerKernels,
    policy: Layouts,
    precision: Precision,
    /// The rows of a group of prompt rows.
    prompt_rows: u32,
    groups: Vec<Group>,
    decided: Option<Vec<Vec<WeightLayout>>>,
    /// The next group of the second pass.
    next: usize,
    probe: Option<Probe>,
    measured_types: Vec<GgmlType>,
    timed: HashMap<(Shape, [u32; 2]), Timed>,
    /// Milliseconds of preparing rows of a width in each form
    /// (`[int8, f16]`), by how, width and rows.
    preps: HashMap<(Prep, u32, u32), [f32; 2]>,
    report: LayoutReport,
}

impl<'a> Chooser<'a> {
    /// A chooser for matrices on `device`, whose products run with
    /// `precision`; `prompt_rows` is the number of rows prompts are read in
    /// at a time.
    pub fn new(
        device: &'a Device,
        kernels: &'a Kernels,
        layers: &'a LayerKernels,
        policy: Layouts,
        precision: Precision,
        prompt_rows: u32,
    ) -> Self {
        Self {
            device,
            kernels,
            layers,
            policy,
            precision,
            prompt_rows: prompt_rows.clamp(1, PROBE_PROMPT_ROWS),
            groups: Vec::new(),
            decided: None,
            next: 0,
            probe: None,
            measured_types: Vec::new(),
            timed: HashMap::new(),
            preps: HashMap::new(),
            report: LayoutReport::default(),
        }
    }

    /// Counts a matrix that was placed in `layout` (in the second pass).
    pub fn placed(&mut self, ty: GgmlType, layout: WeightLayout, n_out: u32, n_in: u32) {
        if self.decided.is_none() {
            return;
        }
        let bytes = |l| Kernels::layout_bytes(ty, l, n_out, n_in).unwrap_or(0);
        let i = usize::from(layout == WeightLayout::Device);
        self.report.matrices[i] += 1;
        self.report.bytes[i] += bytes(layout);
        self.report.padding += bytes(layout).saturating_sub(bytes(WeightLayout::File));
    }

    /// The layouts for the matrices of `shapes`, which read the same input
    /// (prepared as `prep` says) and which only products read. `half` names
    /// the half block of the model they belong to, where the devices of a
    /// tensor split meet after every half.
    pub fn group(
        &mut self,
        shapes: &[Shape],
        usage: Usage,
        prep: Prep,
        half: Option<u32>,
    ) -> Result<Vec<WeightLayout>, ModelError> {
        let Some(decided) = &self.decided else {
            self.groups.push(Group { shapes: shapes.to_vec(), usage, prep, half });
            return Ok(vec![WeightLayout::File; shapes.len()]);
        };
        let i = self.next;
        self.next += 1;
        match (self.groups.get(i), decided.get(i)) {
            (Some(group), Some(layouts)) if group.shapes == shapes => Ok(layouts.clone()),
            _ => Err(ModelError::Invalid(
                "the tensors of a device were laid out differently the second time".into(),
            )),
        }
    }

    fn eligible(&self, shape: Shape) -> bool {
        let (ty, n_out, n_in) = shape;
        n_out > 0 && self.kernels.layout_for(ty, n_in) == WeightLayout::Device
    }

    /// Every group with its matrices in `layout` where they can have it.
    fn all(&self, layout: WeightLayout) -> Vec<Vec<WeightLayout>> {
        let of = |s: &Shape| if self.eligible(*s) { layout } else { WeightLayout::File };
        self.groups.iter().map(|g| g.shapes.iter().map(of).collect()).collect()
    }

    /// Decides the layouts of the groups noted so far: see the module.
    /// `weight_bytes` is what the device's weights will take; `balance` is
    /// where the devices of a tensor split tell each other their times, and
    /// this device's index there.
    pub fn decide(
        &mut self,
        weight_bytes: u64,
        balance: Option<(&Balance, usize)>,
    ) -> Result<(), ModelError> {
        // The others do not wait for a device that measures nothing.
        let nothing_to_tell = || {
            if let Some((balance, index)) = balance {
                balance.exchange(index, Told::default());
            }
        };
        if let Err(error) = self.measure_paths() {
            nothing_to_tell();
            return Err(error);
        }
        let prompt_weight = match self.policy {
            Layouts::File => {
                nothing_to_tell();
                self.decided = Some(self.all(WeightLayout::File));
                return Ok(());
            }
            Layouts::Device => {
                nothing_to_tell();
                self.tune_launches()?;
                self.decided = Some(self.all(WeightLayout::Device));
                return Ok(());
            }
            // Exact products are the reference: every matrix that can takes
            // the device layout and nothing is measured, so that two loads
            // of a model compute the same bits.
            Layouts::Measured { .. } if self.precision == Precision::Exact => {
                nothing_to_tell();
                self.decided = Some(self.all(WeightLayout::Device));
                return Ok(());
            }
            Layouts::Measured { prompt_weight } => f64::from(prompt_weight.max(0.0)),
        };

        // The probes of every group, on weights as large as the device's.
        let free = self.device.mem_info()?.free;
        let l2 = u64::from(self.device.info().l2_cache_bytes);
        let least = PROBE_WEIGHT_BYTES.max(2 * l2);
        let room = weight_bytes.min(free.saturating_sub(PROBE_HEADROOM_BYTES)).max(least);
        let groups = self.groups.clone();
        let mut costed = Vec::with_capacity(groups.len());
        let mut outcome = Ok(());
        for group in &groups {
            match self.cost(group, prompt_weight, room) {
                Ok(c) => costed.push(c),
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        // The probe's memory is free again before anyone waits or places.
        if let Some(probe) = self.probe.take() {
            outcome = outcome.and(probe.stream.synchronize().map_err(ModelError::from));
        }
        if let Err(error) = outcome {
            nothing_to_tell();
            return Err(error);
        }

        // What a group of prompt rows costs this device in the file's layout,
        // all half blocks together, and the slowest other device relative to
        // it: how far this device's time may differ before prompts wait for
        // it.
        let probes = &self.report.probes;
        let mut own = Told { prompt: 0.0, known: true };
        for (group, c) in groups.iter().zip(&costed) {
            if group.half.is_some() {
                let file = c.times(probes, c.candidates.last().expect("the file's layout"));
                own.prompt += file[1] as f32;
            }
        }
        let others: Vec<Told> = match balance {
            Some((balance, index)) => balance
                .exchange(index, own.clone())
                .map(|all| {
                    let others = all.into_iter().enumerate().filter(|(i, _)| *i != index);
                    others.map(|(_, told)| told).filter(|told| told.known).collect()
                })
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let slowest_prompt = others.iter().map(|o| o.prompt).fold(0.0f32, f32::max);
        let prompt_slack =
            if own.prompt > 0.0 { f64::from(slowest_prompt / own.prompt) } else { 0.0 };

        // The groups of a half block are decided together, the others alone.
        let mut units: BTreeMap<(u8, u32), Vec<usize>> = BTreeMap::new();
        for (i, group) in groups.iter().enumerate() {
            let key = match group.half {
                Some(half) => (0, half),
                None => (1, i as u32),
            };
            units.entry(key).or_default().push(i);
        }
        let mut decided: Vec<Vec<WeightLayout>> = vec![Vec::new(); groups.len()];
        let (mut decode_ms, mut prompt_ms) = ([0.0f64; 2], [0.0f64; 2]);
        for ((kind, _), members) in units {
            let slack = match kind {
                0 => prompt_slack,
                _ => 0.0,
            };
            // Every combination of the members' candidates, the ones with
            // the most matrices in the device layout first.
            let mut picks: Vec<Vec<usize>> = vec![Vec::new()];
            for &m in &members {
                let n = costed[m].candidates.len();
                picks = picks
                    .into_iter()
                    .flat_map(|pick| (0..n).map(move |c| [pick.as_slice(), &[c]].concat()))
                    .collect();
            }
            let in_device = |pick: &Vec<usize>| -> usize {
                let of = |(m, c): (&usize, &usize)| {
                    let layouts = &costed[*m].candidates[*c];
                    layouts.iter().filter(|l| **l == WeightLayout::Device).count()
                };
                members.iter().zip(pick).map(of).sum()
            };
            picks.sort_by_key(|pick| std::cmp::Reverse(in_device(pick)));
            let times = |pick: &[usize]| -> [f64; 2] {
                let mut sum = [0.0f64; 2];
                for (m, c) in members.iter().zip(pick) {
                    let t = costed[*m].times(probes, &costed[*m].candidates[*c]);
                    sum = [sum[0] + t[0], sum[1] + t[1]];
                }
                sum
            };
            let file: Vec<usize> =
                members.iter().map(|m| costed[*m].candidates.len() - 1).collect();
            let reference = times(&file);
            // (cost, prompt fraction) of every combination.
            let costs: Vec<(f64, f64)> = picks
                .iter()
                .map(|pick| {
                    let ms = times(pick);
                    let cost = relative_cost(ms, reference, slack, prompt_weight);
                    (cost, relative_cost([0.0, ms[1]], [0.0, reference[1]], slack, 1.0))
                })
                .collect();
            let cheapest = costs.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
            // Of the combinations that are as good, the one that reads
            // prompts fastest; of equals the first, which has the most
            // matrices in the device layout.
            let mut best = 0;
            let mut best_prompt = f64::INFINITY;
            for (i, (cost, prompt)) in costs.iter().enumerate() {
                if *cost <= cheapest + AS_GOOD_WITHIN && *prompt < best_prompt - 1e-9 {
                    (best, best_prompt) = (i, *prompt);
                }
            }
            let chosen = times(&picks[best]);
            let finite = |ms: f64| if ms.is_finite() { ms } else { 0.0 };
            decode_ms = [decode_ms[0] + finite(reference[0]), decode_ms[1] + finite(chosen[0])];
            prompt_ms = [prompt_ms[0] + finite(reference[1]), prompt_ms[1] + finite(chosen[1])];
            for (m, c) in members.iter().zip(&picks[best]) {
                decided[*m] = costed[*m].candidates[*c].clone();
            }
        }
        for (c, layouts) in costed.iter().zip(&decided) {
            for (t, layout) in c.timed.iter().zip(layouts) {
                let i = usize::from(*layout == WeightLayout::Device);
                self.report.probes[t.probe].matrices[i] += 1;
            }
        }
        self.report.balanced_with = others.len();
        self.report.decode_ms = decode_ms.map(|ms| ms as f32);
        self.report.prompt_ms = prompt_ms.map(|ms| ms as f32);
        self.decided = Some(decided);
        Ok(())
    }

    /// Which path of the tile kernels each type of the groups takes in the
    /// file's layout, measured once per type before anything is probed (the
    /// probes of the file's layout run with it). Exact products take the exact
    /// path and measure nothing.
    fn measure_paths(&mut self) -> Result<(), ModelError> {
        if self.precision != Precision::Fast || !self.kernels.has_prefill_kernels() {
            return Ok(());
        }
        let begun = Instant::now();
        let mut types: Vec<GgmlType> =
            self.groups.iter().flat_map(|g| g.shapes.iter().map(|s| s.0)).collect();
        types.sort_by_key(|t| t.name());
        types.dedup();
        for ty in types {
            if self.kernels.has_prefill_path(ty, PrefillPath::F16) {
                self.report.paths.push(self.kernels.measure_prefill_path(ty)?);
            }
        }
        self.report.tables += begun.elapsed();
        Ok(())
    }

    /// How the few-row kernels launch for every shape that can have the device
    /// layout, when nothing else is measured (a matter of speed only: none of the
    /// launches changes a bit of a product).
    fn tune_launches(&self) -> Result<(), ModelError> {
        if self.precision != Precision::Fast {
            return Ok(());
        }
        for &(ty, n_out, n_in) in self.groups.iter().flat_map(|g| g.shapes.iter()) {
            if self.eligible((ty, n_out, n_in)) {
                self.kernels.tune_layout(ty, n_out, n_in)?;
            }
        }
        Ok(())
    }

    /// Probes the matrices of a group and the preparation of its input, and
    /// lists the assignments to choose from.
    fn cost(&mut self, group: &Group, prompt_weight: f64, room: u64) -> Result<Costed, ModelError> {
        let prompt = group.usage.prompt;
        let rows = [group.usage.decode_rows.max(1), if prompt { self.prompt_rows } else { 0 }];
        let mut timed = Vec::with_capacity(group.shapes.len());
        for &shape in &group.shapes {
            timed.push(self.timed(shape, rows, room)?);
        }
        let n_in = group.shapes.first().map_or(0, |s| s.2);
        let prep_ms = [
            self.prep(group.prep, n_in, rows[0], room)?,
            self.prep(group.prep, n_in, rows[1], room)?,
        ];
        let probes = &self.report.probes;
        let all = |layout: WeightLayout| -> Vec<WeightLayout> {
            let of = |t: &Timed| if t.eligible { layout } else { WeightLayout::File };
            timed.iter().map(of).collect()
        };
        // Each matrix in the layout that is cheaper for itself.
        let own_best: Vec<WeightLayout> = timed
            .iter()
            .map(|t| {
                let p = &probes[t.probe];
                let (device, file) = (p.device_ms.map(f64::from), p.file_ms.map(f64::from));
                let cost = relative_cost(device, file, 0.0, prompt_weight);
                let as_file = relative_cost(file, file, 0.0, prompt_weight);
                match t.eligible && cost <= as_file + AS_GOOD_WITHIN {
                    true => WeightLayout::Device,
                    false => WeightLayout::File,
                }
            })
            .collect();
        let mut candidates = vec![all(WeightLayout::Device)];
        for candidate in [own_best, all(WeightLayout::File)] {
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
        // The file's layout is the last, also when it is the only one.
        let file = all(WeightLayout::File);
        candidates.retain(|c| *c != file);
        candidates.push(file);
        Ok(Costed { timed, rows, prep_ms, per_form: group.prep == Prep::Launch, candidates })
    }

    /// The probe of a shape at `rows`: measured once.
    fn timed(&mut self, shape: Shape, rows: [u32; 2], room: u64) -> Result<Timed, ModelError> {
        if let Some(t) = self.timed.get(&(shape, rows)) {
            return Ok(*t);
        }
        let (ty, n_out, n_in) = shape;
        let eligible = self.eligible(shape);
        if eligible && !self.measured_types.contains(&ty) {
            // Which kernel of the device layout serves which number of rows
            // on this device: the products below are launched by that table.
            let begun = Instant::now();
            if !self.kernels.layout_table(ty)?.measured {
                self.kernels.measure_layout(ty)?;
            }
            self.measured_types.push(ty);
            self.report.tables += begun.elapsed();
        }
        let begun = Instant::now();
        let (file_ms, device_ms) = self.time(ty, n_out, n_in, rows, eligible, room)?;
        self.report.probes_time += begun.elapsed();
        let mut file_f16 = [false; 2];
        for (f16, n) in file_f16.iter_mut().zip(rows) {
            let batched = decode_in_batches(self.precision, std::iter::once(ty), n);
            *f16 = n > 0
                && !batched
                && matches!(
                    file_path(self.kernels, self.precision, ty, n)?,
                    Some(PrefillPath::F16 | PrefillPath::F16Flush)
                );
        }
        let t = Timed { probe: self.report.probes.len(), eligible, file_f16 };
        self.report.probes.push(LayoutProbe {
            ty,
            n_out,
            n_in,
            decode_rows: rows[0],
            prompt_rows: rows[1],
            file_ms,
            device_ms,
            matrices: [0; 2],
        });
        self.timed.insert((shape, rows), t);
        Ok(t)
    }

    /// The probe's stream, clock and buffers, made when first needed, with
    /// `room` bytes of weights.
    fn probe<'p>(
        probe: &'p mut Option<Probe>,
        device: &Device,
        room: u64,
        report: &mut LayoutReport,
    ) -> Result<&'p mut Probe, ModelError> {
        if probe.is_none() {
            let stream = Stream::new(device, 0)?;
            let w = DeviceBuffer::new(device, room)?;
            // What the buffers hold does not matter to the time of a product
            // (no kernel of these types branches on or addresses by a value)
            // as long as it is numbers: small codes and scales here,
            // activations near 0.01 below.
            // SAFETY: the whole buffer.
            unsafe { stream.fill(w.ptr(), 0x11, room as usize)? };
            report.probe_bytes = room;
            *probe = Some(Probe {
                stream,
                start: Event::timed(device)?,
                end: Event::timed(device)?,
                w,
                cursor: 0,
                x: Grown::default(),
                act_q: Grown::default(),
                act_h: Grown::default(),
                y: Grown::default(),
            });
        }
        Ok(probe.as_mut().expect("made above"))
    }

    /// Milliseconds of preparing `n` rows of `n_in` values as 8-bit
    /// activations and as f16, as `prep` does it: measured once per width
    /// and number of rows.
    fn prep(&mut self, prep: Prep, n_in: u32, n: u32, room: u64) -> Result<[f32; 2], ModelError> {
        if n == 0 || n_in == 0 || prep == Prep::Merge {
            return Ok([0.0; 2]);
        }
        if let Some(ms) = self.preps.get(&(prep, n_in, n)) {
            return Ok(*ms);
        }
        let begun = Instant::now();
        let (device, kernels, layers) = (self.device, self.kernels, self.layers);
        let f16_kernels = kernels.has_prefill_kernels();
        let probe = Self::probe(&mut self.probe, device, room, &mut self.report)?;
        let x_bytes = u64::from(n) * u64::from(n_in) * 4;
        let fresh = probe.x.len() < x_bytes;
        let x = probe.x.ensure(device, x_bytes)?;
        let values = u64::from(n) * u64::from(n_in);
        let act_q = probe.act_q.ensure(device, Kernels::quantized_activation_bytes(n_in, n))?;
        let act_h =
            probe.act_h.ensure(device, Kernels::activation_bytes(ActivationForm::F16, n_in, n))?;
        if fresh {
            // SAFETY: the whole buffer.
            unsafe { probe.stream.fill(x, 0x3c, probe.x.len() as usize)? };
        }
        // The f32 copy that the quantizing kernels leave, for a row too wide
        // for the fused kernel that writes f16.
        let copy = act_q.offset(ares_cuda::quantized_activation_bytes(values) - values * 4);
        let (int8, f16) = (ActivationForm::Int8, ActivationForm::F16);
        let gate = OutputGate::Silu;
        let launch = |stream: &Stream, as_f16: bool| -> Result<(), ModelError> {
            // SAFETY: the probe's buffers hold `n` rows of `n_in` values and
            // either form of them; the inputs of the fused kernels are the
            // same rows, read only.
            unsafe {
                match (prep, as_f16) {
                    (Prep::Merge, _) => {}
                    (Prep::Launch, false) => {
                        kernels.prepare_activations(stream, int8, x, act_q, n_in, n)?;
                    }
                    (Prep::Launch, true) => {
                        kernels.prepare_activations(stream, f16, x, act_h, n_in, n)?;
                    }
                    (Prep::GatedNorm { heads, width }, as_f16) => {
                        let fused = heads <= LayerKernels::GATED_NORM_F16_MAX_HEADS;
                        if as_f16 && fused {
                            layers.gated_norm_f16(
                                stream, gate, x, x, x, act_h, n, heads, width, 1e-6,
                            )?;
                        } else {
                            let items = n * heads;
                            layers.gated_norm_quant(
                                stream, gate, x, x, x, act_q, items, width, 1e-6,
                            )?;
                        }
                        if as_f16 && !fused {
                            kernels.prepare_activations(stream, f16, copy, act_h, n_in, n)?;
                        }
                    }
                    (Prep::Swiglu, as_f16) => {
                        let fused = n_in <= LayerKernels::SWIGLU_F16_MAX_WIDTH;
                        if as_f16 && fused {
                            layers.swiglu_f16(stream, x, x, act_h, n, n_in)?;
                        } else {
                            layers.swiglu_quant(stream, x, x, act_q, n * n_in)?;
                        }
                        if as_f16 && !fused {
                            kernels.prepare_activations(stream, f16, copy, act_h, n_in, n)?;
                        }
                    }
                }
            }
            Ok(())
        };
        let mut best = [f32::INFINITY; 2];
        for round in 0..=ROUNDS {
            for (i, as_f16) in [false, true].into_iter().enumerate() {
                if as_f16 && !f16_kernels {
                    continue;
                }
                let ms = probe.run(BURST_FEW, |probe| launch(&probe.stream, as_f16))?;
                if round > 0 {
                    best[i] = best[i].min(ms);
                }
            }
        }
        self.preps.insert((prep, n_in, n), best);
        if prep == Prep::Launch {
            self.report.preparations.push((n_in, n, best));
        }
        self.report.probes_time += begun.elapsed();
        Ok(best)
    }

    /// Milliseconds of the product of a matrix of this type and shape with
    /// each of `rows` (0: not measured) in the file's layout and in the
    /// device layout (the file's again for a matrix that cannot have it).
    fn time(
        &mut self,
        ty: GgmlType,
        n_out: u32,
        n_in: u32,
        rows: [u32; 2],
        eligible: bool,
        room: u64,
    ) -> Result<([f32; 2], [f32; 2]), ModelError> {
        let (device, kernels, precision) = (self.device, self.kernels, self.precision);
        let invalid = || ModelError::Invalid(format!("{ty} has no rows of {n_in} values"));
        let row_bytes =
            Kernels::layout_bytes(ty, WeightLayout::File, 1, n_in).ok_or_else(invalid)?;
        let most_rows = rows[0].max(rows[1]);
        let full_rows = n_out;
        let l2 = u64::from(device.info().l2_cache_bytes);
        // The rows of the probe matrix: the matrix's, as far as the bounds allow.
        let n_out = n_out
            .min((PROBE_WEIGHT_BYTES.max(l2 * 5 / 4) / row_bytes.max(1)) as u32)
            .min((PROBE_RESULT_BYTES / (4 * u64::from(most_rows))) as u32)
            .max(16);
        let w_bytes = u64::from(n_out.next_multiple_of(16)) * row_bytes;
        let f16_kernels = kernels.has_prefill_kernels();
        let probe = Self::probe(&mut self.probe, device, room, &mut self.report)?;
        if eligible {
            // How the few-row kernels launch for this shape and for the probe matrix, which
            // may be cut (a matter of speed only: none changes a bit of the product), over
            // the probe's weights, so that the products below are timed as they will run.
            kernels.tune_layout_over(ty, full_rows, n_in, &probe.w)?;
            kernels.tune_layout_over(ty, n_out, n_in, &probe.w)?;
        }
        let x_bytes = u64::from(most_rows) * u64::from(n_in) * 4;
        let fresh_x = probe.x.len() < x_bytes;
        let x = probe.x.ensure(device, x_bytes)?;
        // With room for the padding between the launches of the decode
        // kernels in batches.
        let batches = u64::from(most_rows.div_ceil(MAX_BATCH));
        let act_q = probe.act_q.ensure(
            device,
            Kernels::quantized_activation_bytes(n_in, most_rows) + 64 * batches + 256,
        )?;
        let act_h = probe
            .act_h
            .ensure(device, Kernels::activation_bytes(ActivationForm::F16, n_in, most_rows))?;
        let y = probe.y.ensure(device, u64::from(most_rows) * u64::from(n_out) * 4)?;
        if fresh_x {
            // SAFETY: the whole buffer.
            unsafe { probe.stream.fill(x, 0x3c, probe.x.len() as usize)? };
        }

        let (mut file_ms, mut device_ms) = ([0.0f32; 2], [0.0f32; 2]);
        for (i, n) in rows.into_iter().enumerate() {
            if n == 0 {
                continue;
            }
            let burst = if n <= 2 * MAX_BATCH { BURST_FEW } else { BURST_MANY };
            // The file's layout: the decode kernels in one launch or several,
            // or the tile kernels, as a step multiplies it.
            let batched = decode_in_batches(precision, std::iter::once(ty), n);
            let path = file_path(kernels, precision, ty, n)?;
            let per = if batched { MAX_BATCH } else { n };
            let stride = Kernels::quantized_activation_bytes(n_in, per).next_multiple_of(64);
            let ranges = move || {
                (0..n.div_ceil(per)).map(move |b| {
                    (b * per, per.min(n - b * per), act_q.offset(u64::from(b) * stride))
                })
            };
            let file_runs =
                if batched || path.is_none() { Kernels::has_fast_matvec(ty) } else { f16_kernels };
            // SAFETY: the probe's buffers hold `most_rows` rows of `n_in`
            // values in every form and their products with `n_out` rows.
            unsafe {
                let stream = &probe.stream;
                if file_runs {
                    match path {
                        _ if batched => {
                            for (first, count, act) in ranges() {
                                let at = x.offset(u64::from(first) * u64::from(n_in) * 4);
                                kernels.quantize_activations(stream, at, act, n_in, count)?;
                            }
                        }
                        None => kernels.quantize_activations(stream, x, act_q, n_in, n)?,
                        Some(PrefillPath::Int8) => kernels.prepare_prefill_activations(
                            stream,
                            PrefillPath::Int8,
                            x,
                            act_q,
                            n_in,
                            n,
                        )?,
                        Some(PrefillPath::F16 | PrefillPath::F16Flush) => {}
                    }
                }
                if f16_kernels {
                    kernels.prepare_activations(stream, ActivationForm::F16, x, act_h, n_in, n)?;
                }
            }
            let file = |stream: &Stream, w: DevicePtr| -> Result<(), ModelError> {
                // SAFETY: as above; the activations were prepared for the path.
                unsafe {
                    match path {
                        _ if batched => {
                            for (first, count, act) in ranges() {
                                let at = y.offset(u64::from(first) * u64::from(n_out) * 4);
                                kernels.matvec(stream, ty, w, act, at, n_in, n_out, count)?;
                            }
                        }
                        None => kernels.matvec(stream, ty, w, act_q, y, n_in, n_out, n)?,
                        Some(path) => {
                            let act = match path {
                                PrefillPath::Int8 => act_q,
                                PrefillPath::F16 | PrefillPath::F16Flush => act_h,
                            };
                            kernels.matmul(stream, ty, path, w, act, y, n_in, n_out, n)?;
                        }
                    }
                }
                Ok(())
            };
            let rearranged = |stream: &Stream, w: DevicePtr| -> Result<(), ModelError> {
                // SAFETY: as above.
                unsafe {
                    kernels.product(
                        stream,
                        ty,
                        WeightLayout::Device,
                        precision,
                        w,
                        act_h,
                        y,
                        n_in,
                        n_out,
                        n,
                    )?;
                }
                Ok(())
            };
            // One layout's runs after the other's, and of each the fastest but
            // the first, which warms the clocks up. (Runs of the two layouts
            // in turn measured the file's layout 2 to 4 microseconds slower
            // at one row than runs of its own, on both cards, and the model's
            // steps agree with the latter.)
            let mut best_file = f32::INFINITY;
            for round in 0..=ROUNDS {
                if !file_runs {
                    break;
                }
                let ms = probe.run(burst, |probe| {
                    let w = probe.place(w_bytes);
                    file(&probe.stream, w)
                })?;
                if round > 0 {
                    best_file = best_file.min(ms);
                }
            }
            let mut best_device = if eligible { f32::INFINITY } else { best_file };
            for round in 0..=ROUNDS {
                if !eligible {
                    break;
                }
                let ms = probe.run(burst, |probe| {
                    let w = probe.place(w_bytes);
                    rearranged(&probe.stream, w)
                })?;
                if round > 0 {
                    best_device = best_device.min(ms);
                }
            }
            (file_ms[i], device_ms[i]) = (best_file, best_device);
        }
        Ok((file_ms, device_ms))
    }

    /// Ends the choice and returns what was done.
    pub fn finish(self) -> LayoutReport {
        self.report
    }
}

impl Costed {
    /// The milliseconds of an assignment at the decode rows and at the
    /// prompt rows: its products, and what preparing the forms costs that
    /// its matrices read.
    fn times(&self, probes: &[LayoutProbe], layouts: &[WeightLayout]) -> [f64; 2] {
        std::array::from_fn(|r| {
            if self.rows[r] == 0 {
                return 0.0;
            }
            let (mut ms, mut int8, mut f16) = (0.0f64, false, false);
            for (t, layout) in self.timed.iter().zip(layouts) {
                let probe = &probes[t.probe];
                match layout {
                    WeightLayout::Device => {
                        ms += f64::from(probe.device_ms[r]);
                        f16 = true;
                    }
                    WeightLayout::File => {
                        ms += f64::from(probe.file_ms[r]);
                        match t.file_f16[r] {
                            true => f16 = true,
                            false => int8 = true,
                        }
                    }
                }
            }
            let prep = self.prep_ms[r].map(f64::from);
            ms + match (self.per_form, int8, f16) {
                (true, int8, f16) => {
                    f64::from(u8::from(int8)) * prep[0] + f64::from(u8::from(f16)) * prep[1]
                }
                // One kernel writes the form of its one reader.
                (false, _, true) => prep[1],
                (false, _, false) => prep[0],
            }
        })
    }
}

/// The cost of an assignment whose products take `ms` at the decode rows and
/// at the prompt rows, where every matrix in the file's layout takes
/// `reference`: each time as a fraction of the reference's, the prompts'
/// weighted. `prompt_slack` is the prompt time of the slowest other device as
/// a fraction of the reference (0 without one): the prompt time counts only
/// as far as it is the largest. A regime that is not run (0 ms in the
/// reference) costs nothing, and one that the file's layout has no kernel
/// for costs the same for every assignment that can run it.
fn relative_cost(ms: [f64; 2], reference: [f64; 2], prompt_slack: f64, prompt_weight: f64) -> f64 {
    let fraction = |r: usize, slack: f64| {
        if reference[r] == 0.0 {
            0.0
        } else if reference[r].is_finite() {
            (ms[r] / reference[r]).max(slack) / slack.max(1.0)
        } else if ms[r].is_finite() {
            1.0
        } else {
            f64::INFINITY
        }
    };
    fraction(0, 0.0) + prompt_weight * fraction(1, prompt_slack)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_count_as_far_as_they_wait_for_the_device() {
        let file = [1.0, 10.0];
        // Alone, a percent for a percent: 3 % slower steps for 10 % faster
        // prompts is taken, for 2 % not.
        assert!(relative_cost([1.03, 9.0], file, 0.0, 1.0) < 2.0);
        assert!(relative_cost([1.03, 9.8], file, 0.0, 1.0) > 2.0);
        // Decode steps first: prompts buy nothing.
        assert!(relative_cost([1.03, 5.0], file, 0.0, 0.0) > relative_cost(file, file, 0.0, 0.0));
        // Prompts wait for another device, which takes 1.5 times as long:
        // a gain here is worth nothing, a slower decode step still costs.
        let as_file = relative_cost(file, file, 1.5, 1.0);
        assert_eq!(relative_cost([1.0, 5.0], file, 1.5, 1.0), as_file);
        assert!(relative_cost([1.01, 5.0], file, 1.5, 1.0) > as_file);
        // The other device takes 0.9 times as long: a gain counts down to it.
        let to_theirs = relative_cost([1.0, 9.0], file, 0.9, 1.0);
        assert!(to_theirs < relative_cost(file, file, 0.9, 1.0));
        assert_eq!(relative_cost([1.0, 5.0], file, 0.9, 1.0), to_theirs);
        // A regime the file's layout cannot run does not decide between the
        // assignments that can.
        let none = [1.0, f64::INFINITY];
        assert!(relative_cost([1.0, 3.0], none, 0.0, 1.0).is_finite());
        assert!(relative_cost(none, none, 0.0, 1.0).is_infinite());
    }

    #[test]
    fn matrices_take_the_decode_kernels_in_launches_only_where_those_are_faster() {
        let fast =
            |types: &[GgmlType], n| decode_in_batches(Precision::Fast, types.iter().copied(), n);
        for ty in ares_core::ggml_type::ALL_TYPES.into_iter().filter(|t| t.is_weight()) {
            // One launch takes up to eight rows, so there is nothing to split.
            assert!(!fast(&[ty], 1) && !fast(&[ty], MAX_BATCH), "{ty}");
            for n in [MAX_BATCH + 1, 16, 17, 100, 512] {
                let expected = Kernels::decode_kernel_rows(ty).is_some_and(|limit| n <= limit);
                assert_eq!(fast(&[ty], n), expected, "{ty} with {n} rows");
            }
            // Every weight type has a tile kernel, so none is split for lack of one.
            assert!(!fast(&[ty], 512), "{ty}");
        }
        assert!(!fast(&[GgmlType::IQ2_XXS], 512) && !fast(&[GgmlType::IQ2_XXS], 17));
        assert!(
            fast(&[GgmlType::IQ1_S], 12)
                && !fast(&[GgmlType::IQ1_S], 13)
                && !fast(&[GgmlType::Q2_K], 9)
        );
        assert!(!fast(&[GgmlType::Q4_K], 17) && fast(&[GgmlType::Q4_K], 16));
        // Matrices that read one input are multiplied the same way: one that is faster on the
        // tile kernels keeps the others off the decode kernels.
        assert!(fast(&[GgmlType::Q4_K, GgmlType::IQ4_XS], 12));
        assert!(!fast(&[GgmlType::Q4_K, GgmlType::Q8_0], 12));
        // The exact products are the tile kernels' f16 path: no launches of decode kernels.
        assert!(!decode_in_batches(Precision::Exact, [GgmlType::IQ2_XXS].into_iter(), 100));
    }

    #[test]
    fn devices_tell_each_other_and_start_anew() {
        let balance = std::sync::Arc::new(Balance::new(2));
        for round in 0..2 {
            let other = balance.clone();
            let told = |ms: f32| Told { prompt: ms, known: true };
            let theirs = 2.0 + round as f32;
            let thread = std::thread::spawn(move || other.exchange(1, told(theirs)));
            let all = balance.exchange(0, told(1.0)).expect("both told");
            assert_eq!(all[1].prompt, theirs);
            assert_eq!(thread.join().unwrap().expect("both told")[0].prompt, 1.0);
        }
    }

    #[test]
    fn a_device_that_gives_up_does_not_keep_the_others_waiting() {
        let balance = std::sync::Arc::new(Balance::new(2));
        let other = balance.clone();
        let begun = Instant::now();
        let waiting =
            std::thread::spawn(move || other.exchange(1, Told { prompt: 2.0, known: true }));
        // The device that fails first tells nothing; the other is let go with
        // what it knows, long before the patience of two minutes.
        std::thread::sleep(Duration::from_millis(50));
        balance.give_up(0);
        let all = waiting.join().unwrap().expect("let go, not timed out");
        assert!(!all[0].known && all[1].known);
        assert!(begun.elapsed() < Duration::from_secs(10));
        // A device that has told is not undone by giving up afterwards.
        let balance = Balance::new(1);
        balance.exchange(0, Told { prompt: 3.0, known: true });
        balance.give_up(0);
    }
}
