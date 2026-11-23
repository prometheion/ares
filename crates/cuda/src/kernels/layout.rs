//! The device layouts of weight matrices: a matrix rearranged once at load, in the
//! bytes of the file's form, into the order the kernels read it, and the one family
//! of kernels (`kernels/mmr.cuh`) that serves every number of activation rows from
//! that one copy.
//!
//! A model holds one buffer per matrix and says in which [`WeightLayout`] it lies. Which
//! prepared form of the activations a product reads, and which kernel computes it,
//! is this module's business ([`Kernels::activation_form`], [`Kernels::product`]):
//! for a matrix in the device layout the few-row kernels up to 16 rows (the rows of a
//! decode step and of a window of drafts; their launch is [`FewPlan`], measured per
//! matrix shape by [`Kernels::tune_layout`]) and a table measured on the device for more
//! ([`Kernels::measure_layout`]), for one in the file's layout the decode kernels up to
//! a few rows and the tile kernels above.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ares_core::GgmlType;

use super::{DEFAULT_SHARED, Kernels, MAX_BATCH, Precision, PrefillPath, check_alignment};
use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::module::{Args, Function, Module};
use crate::stream::{Event, Stream};

/// How the bytes of a weight matrix lie on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightLayout {
    /// As in the model file. Every type has kernels for it: the decode kernels
    /// ([`Kernels::matvec`]) and, for some, the tile kernels ([`Kernels::matmul`]).
    File,
    /// Rearranged by [`Kernels::rearrange`] into the order the tensor cores read it,
    /// in the same number of bytes with the rows rounded up to a multiple of 16. The
    /// types of [`Kernels::has_device_layout`] have it.
    Device,
}

/// The prepared form of the activations a product reads
/// ([`Kernels::prepare_activations`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActivationForm {
    /// int8 with a scale per 32 values ([`Kernels::quantize_activations`]'s format).
    Int8,
    /// f16, each row divided by a power of two that is stored behind the values.
    F16,
}

/// One of the kernels of the family over a device layout for 17 rows and more: the weight
/// rows and activation rows a block computes, its threads, and whether it accumulates in
/// f32.
#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    weight_rows: u32,
    batch_rows: u32,
    threads: u32,
    exact: bool,
}

impl Shape {
    /// The dynamic shared memory of a block (`AresMmrShared` in `kernels/mmr.cuh`): three
    /// stages of the activation rows of a step, each row 16 bytes longer than its 64 f16
    /// values.
    const fn shared_bytes(self) -> u32 {
        3 * self.batch_rows * (64 * 2 + 16)
    }
}

/// The kernels of a layout for 17 rows and more, `ares_mmr_<name>_<type>`
/// (`ARES_MMR_KERNELS` in `kernels/mmr.cuh`), ordered by activation rows.
const SHAPES: [Shape; 4] = [
    Shape { name: "r32f", weight_rows: 128, batch_rows: 32, threads: 128, exact: true },
    Shape { name: "r32h", weight_rows: 128, batch_rows: 32, threads: 128, exact: false },
    Shape { name: "r64f", weight_rows: 128, batch_rows: 64, threads: 128, exact: true },
    Shape { name: "r64h", weight_rows: 128, batch_rows: 64, threads: 128, exact: false },
];

/// The most rows the few-row kernels take: a decode step and a window of drafts. Their
/// arithmetic is one, whatever the number of rows and the launch ([`FewPlan`]); with more
/// rows the table below serves.
pub const FEW_ROWS: u32 = 16;
/// The numbers of activation rows the table distinguishes above [`FEW_ROWS`]: up to 32 rows,
/// and more.
const BUCKETS: [u32; 2] = [32, u32::MAX];
/// The rows each bucket is measured with. The last is two activation tiles of the widest
/// shape, so that it runs grouped as a prompt does.
const PROBE_ROWS: [u32; 2] = [32, 128];
/// The shapes that can serve each bucket in one activation tile, narrowest first; the next
/// wider ones are measured with it, since a wider block is sometimes the faster one.
const CANDIDATES: [&[usize]; 2] = [&[0, 1, 2, 3], &[2, 3]];
/// A kernel within this factor of the fastest of its bucket is as good as it: the
/// differences below that are the probe's noise and its shape. Of the kernels that are as
/// good, [`Precision::Fast`] takes one with f32 accumulation (exact arithmetic where it
/// costs nothing) and the narrowest: it has the most blocks on the device at once, so the
/// fewest matrices leave blocks over that run alone at the end (`kernels/mmr.cuh`).
const AS_GOOD_WITHIN: f32 = 1.05;
/// Activation tiles that take a weight tile together (`group` of the kernels) share its
/// weights through L2 and must keep their own activations there too: a group is as many
/// tiles as put their activations into this share of the device's L2 cache, within the
/// bounds below. On an RTX 4070 (36 MiB of L2) that is 8 to 13 tiles of 64 rows of 5120
/// values, where 8 and 16 measured 87 TFLOPS against 84 for 2; on an RTX 3060 (3 MiB) it is
/// 2, which measured the same as 8 and better than 16.
const GROUP_L2_SHARE: u64 = 4;
const GROUP_MIN: u64 = 2;
const GROUP_MAX: u64 = 16;

/// The types with a device layout and their kernels' name part (`ARES_MMR_CODES` in
/// `kernels/prefill.cu`). All of them step through a row in super-blocks of 256
/// values, so a row must be a whole number of those.
const TYPES: [(GgmlType, &str); 12] = [
    (GgmlType::Q4_K, "q4_K"),
    (GgmlType::Q5_K, "q5_K"),
    (GgmlType::Q6_K, "q6_K"),
    (GgmlType::Q3_K, "q3_K"),
    (GgmlType::Q2_K, "q2_K"),
    (GgmlType::Q4_0, "q4_0"),
    (GgmlType::Q4_1, "q4_1"),
    (GgmlType::Q5_0, "q5_0"),
    (GgmlType::Q5_1, "q5_1"),
    (GgmlType::Q8_0, "q8_0"),
    (GgmlType::IQ4_XS, "iq4_xs"),
    (GgmlType::IQ4_NL, "iq4_nl"),
];
const COLUMNS_MULTIPLE: u32 = 256;

/// One kernel of a table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutKernel {
    /// The shape's name in `kernels/mmr.cuh`.
    pub name: &'static str,
    /// The weight rows and the activation rows one thread block computes.
    pub weight_rows: u32,
    pub batch_rows: u32,
    /// Whether it accumulates in f32 (the exact product of the f16 operands) or in
    /// f16 added to f32 every 64 values.
    pub f32_accumulation: bool,
    /// The time of the probe product, in milliseconds; 0 in a table that was not
    /// measured.
    pub probe_ms: f32,
}

/// The kernels of one bucket of rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutRows {
    /// The most activation rows of the bucket (`u32::MAX` for the last).
    pub up_to_rows: u32,
    /// The kernel of [`Precision::Fast`] and of [`Precision::Exact`].
    pub fast: LayoutKernel,
    pub exact: LayoutKernel,
}

/// Which kernel serves which number of activation rows above [`FEW_ROWS`] for one type on
/// one device.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutTable {
    pub ty: GgmlType,
    /// Whether the table was measured on this device ([`Kernels::measure_layout`]).
    /// Otherwise it is the one kept from the cards the kernels were written on: f32
    /// accumulation where the tensor cores run it at full rate, f16 accumulation where
    /// they do not.
    pub measured: bool,
    pub rows: [LayoutRows; 2],
    /// The shape of each bucket, `[fast, exact]`.
    picks: [[usize; 2]; 2],
}

impl LayoutTable {
    fn new(ty: GgmlType, measured: bool, picks: [[usize; 2]; 2], ms: [[f32; 2]; 2]) -> Self {
        let kernel = |shape: usize, probe_ms: f32| LayoutKernel {
            name: SHAPES[shape].name,
            weight_rows: SHAPES[shape].weight_rows,
            batch_rows: SHAPES[shape].batch_rows,
            f32_accumulation: SHAPES[shape].exact,
            probe_ms,
        };
        let rows = std::array::from_fn(|b| LayoutRows {
            up_to_rows: BUCKETS[b],
            fast: kernel(picks[b][0], ms[b][0]),
            exact: kernel(picks[b][1], ms[b][1]),
        });
        Self { ty, measured, rows, picks }
    }

    /// The table of a device that was not measured.
    fn kept(ty: GgmlType, full_rate: bool) -> Self {
        let wide = if full_rate { [0, 2] } else { [1, 3] };
        Self::new(ty, false, [[wide[0], 0], [wide[1], 2]], [[0.0; 2]; 2])
    }

    fn shape(&self, precision: Precision, n_rows: u32) -> Shape {
        let bucket = BUCKETS.iter().position(|top| n_rows <= *top).unwrap_or(BUCKETS.len() - 1);
        SHAPES[self.picks[bucket][usize::from(precision == Precision::Exact)]]
    }
}

/// The parts K is cut into by the few-row kernels (`ARES_MMF_PARTS`): their arithmetic.
const FEW_PARTS: u32 = 8;

/// How a few-row kernel is launched: the warps of a thread block, and how many of them
/// compute one tile of 16 weight rows, each over its share of the `FEW_PARTS` parts of K.
/// Which is the fastest depends on the shape of the matrix and the device (a matrix of a
/// few tiles wants all of the device's warps on them, one of many tiles wants no sum
/// between warps); none changes a bit of the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FewPlan {
    /// Warps in a block: 4 or 8.
    pub block_warps: u32,
    /// The base-2 logarithm of the warps that compute a tile, at most that of the block
    /// and of `FEW_PARTS`.
    pub log2_warps_per_tile: u32,
}

impl FewPlan {
    /// The launches that are measured, narrowest block first.
    const ALL: [FewPlan; 4] = [
        FewPlan { block_warps: 4, log2_warps_per_tile: 0 },
        FewPlan { block_warps: 4, log2_warps_per_tile: 1 },
        FewPlan { block_warps: 4, log2_warps_per_tile: 2 },
        FewPlan { block_warps: 8, log2_warps_per_tile: 3 },
    ];

    /// Every launch the few-row kernels have.
    pub fn all() -> [FewPlan; 4] {
        Self::ALL
    }

    /// What a matrix of `tiles` tiles of 16 weight rows gets before
    /// [`Kernels::tune_layout`] has measured it, on a device that keeps `device_warps` warps
    /// of these kernels resident at once: as many warps on a tile (1, 2, 4 or 8) as the
    /// device can still hold for all the tiles.
    pub fn untuned(tiles: u32, device_warps: u32) -> Self {
        let mut log2 = 0;
        while log2 < 3 && u64::from(tiles) << (log2 + 1) <= u64::from(device_warps) {
            log2 += 1;
        }
        Self::ALL[log2]
    }

    fn block(self) -> usize {
        usize::from(self.block_warps == 8)
    }

    fn is_valid(self) -> bool {
        Self::ALL.contains(&self)
    }

    /// The dynamic shared memory of a block (`ares_mmf_smem_bytes` in `kernels/mmr.cuh`):
    /// the parts of K the warps of a tile hand to each other.
    const fn shared_bytes(self, nb: u32) -> u32 {
        if self.log2_warps_per_tile == 0 {
            0
        } else {
            (self.block_warps >> self.log2_warps_per_tile) * FEW_PARTS * nb * 4 * 32 * 4
        }
    }
}

/// The kernels of one type with a device layout.
#[derive(Debug, Clone)]
struct TypeKernels {
    ty: GgmlType,
    repack: Function,
    shapes: [Function; 4],
    /// The few-row kernels, `[rows 8 or 16][block of 4 or 8 warps]`.
    few: [[Function; 2]; 2],
    /// The warps of the few-row kernels one multiprocessor keeps resident at once.
    warps_per_multiprocessor: u32,
}

/// The launches measured so far, by type and the rows and columns of the matrix.
type Plans = HashMap<(GgmlType, u32, u32), FewPlan>;

/// The kernels over the device layouts on one device, and what was measured so far.
#[derive(Debug, Clone)]
pub(super) struct Layouts {
    types: Vec<TypeKernels>,
    tables: Arc<Mutex<Vec<LayoutTable>>>,
    plans: Arc<Mutex<Plans>>,
    /// When the last launch was measured: the clocks of a device that has been idle are low.
    tuned_at: Arc<Mutex<Option<Instant>>>,
}

impl Layouts {
    pub(super) fn load(module: &Module, shared_limit: u32) -> Result<Self> {
        let types = TYPES
            .iter()
            .map(|(ty, suffix)| {
                let mut shapes = Vec::with_capacity(SHAPES.len());
                for shape in SHAPES {
                    let function =
                        module.function(&format!("ares_mmr_{}_{suffix}", shape.name))?;
                    let bytes = shape.shared_bytes();
                    if bytes > shared_limit {
                        return Err(Error::Invalid {
                            what: "layout kernel",
                            detail: format!(
                                "{} needs {bytes} bytes of shared memory per block, the device allows {shared_limit}",
                                shape.name
                            ),
                        });
                    }
                    if bytes > DEFAULT_SHARED {
                        function.allow_dynamic_shared(bytes)?;
                    }
                    shapes.push(function);
                }
                let few = |rows: u32, warps: u32| module.function(&format!("ares_mmf_{rows}_w{warps}_{suffix}"));
                let few = [[few(8, 4)?, few(8, 8)?], [few(16, 4)?, few(16, 8)?]];
                Ok(TypeKernels {
                    ty: *ty,
                    repack: module.function(&format!("ares_repack_{suffix}"))?,
                    shapes: <[Function; 4]>::try_from(shapes).expect("four shapes"),
                    warps_per_multiprocessor: few[0][0].blocks_per_multiprocessor(128, 0)? * 4,
                    few,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { types, tables: Arc::default(), plans: Arc::default(), tuned_at: Arc::default() })
    }

    fn of(&self, ty: GgmlType) -> Result<&TypeKernels> {
        self.types.iter().find(|t| t.ty == ty).ok_or_else(|| no_layout(ty))
    }
}

fn no_layout(ty: GgmlType) -> Error {
    Error::Invalid {
        what: "device layout",
        detail: format!(
            "{ty} has none (the types that have one: {})",
            TYPES.map(|(t, _)| t.name()).join(", ")
        ),
    }
}

/// The group of activation tiles that take every weight tile in turn: what
/// [`GROUP_L2_SHARE`] allows of tiles of `tile_bytes` of activations, and a divisor
/// of the tiles when one is near, so that every wave of blocks works on whole groups.
/// The tile kernels of the file's layout ([`Kernels::matmul`]) group the same way: on
/// an RTX 4060 Ti (32 MiB of L2) their f16 path at 512 rows (4 tiles) took 51, 55 and 57
/// TFLOPS for Q4_K in groups of 3 (and one tile alone), 2 and 4, and at 1024 rows 57, 55
/// and 58; a fixed 4 MiB had given groups of 3.
pub(super) fn group_of(l2_cache_bytes: u32, tile_bytes: u64, n_batch_tiles: u32) -> u32 {
    let fits = u64::from(l2_cache_bytes) / GROUP_L2_SHARE / tile_bytes.max(1);
    let group = (fits.clamp(GROUP_MIN, GROUP_MAX) as u32).min(n_batch_tiles);
    (1..=group)
        .rev()
        .find(|g| n_batch_tiles.is_multiple_of(*g))
        .filter(|g| *g * 2 > group)
        .unwrap_or(group)
}

impl Kernels {
    /// Whether a matrix of `ty` with rows of `n_in` values has a device layout
    /// ([`WeightLayout::Device`]): the type has one and the rows are whole steps of it.
    pub fn has_device_layout(ty: GgmlType, n_in: u32) -> bool {
        TYPES.iter().any(|(t, _)| *t == ty) && n_in != 0 && n_in.is_multiple_of(COLUMNS_MULTIPLE)
    }

    /// The layout a loader should give a matrix of `ty` with rows of `n_in` values
    /// on this device: the device layout where the type has one, the file's otherwise.
    pub fn layout_for(&self, ty: GgmlType, n_in: u32) -> WeightLayout {
        if Self::has_device_layout(ty, n_in) { WeightLayout::Device } else { WeightLayout::File }
    }

    /// The bytes of a matrix of `n_out` rows of `n_in` values in `layout`; `None` when
    /// the type has no such layout or the rows are not whole blocks. The device
    /// layout takes the bytes of the file's with the rows rounded up to a multiple
    /// of 16.
    pub fn layout_bytes(ty: GgmlType, layout: WeightLayout, n_out: u32, n_in: u32) -> Option<u64> {
        let (values, bytes) = ty.block();
        if !n_in.is_multiple_of(values) {
            return None;
        }
        let row = u64::from(n_in / values) * u64::from(bytes);
        match layout {
            WeightLayout::File => Some(u64::from(n_out) * row),
            WeightLayout::Device => Self::has_device_layout(ty, n_in)
                .then(|| u64::from(n_out.next_multiple_of(16)) * row),
        }
    }

    /// The alignment, in bytes, a matrix of `ty` needs on the device in `layout`.
    pub fn layout_alignment(ty: GgmlType, layout: WeightLayout) -> u64 {
        match layout {
            WeightLayout::Device => 16,
            WeightLayout::File => {
                super::FAST_TYPES.iter().find(|(t, ..)| *t == ty).map_or(16, |(_, _, align)| *align)
            }
        }
    }

    /// Rearranges the matrix at `src` (`n_out` rows of `n_in` values of `ty` in the
    /// file's layout) into its device layout at `dst`. One kernel; it runs once per
    /// matrix, at load.
    ///
    /// A matrix that a tensor split cuts into rows or columns is cut first, in the
    /// file's layout, and each part rearranged on the device that holds it: the
    /// layout mixes 16 rows, so a part cannot be cut out of a rearranged matrix
    /// except at multiples of 16 rows and 256 columns.
    ///
    /// # Safety
    ///
    /// `src` must address the matrix and `dst`
    /// [`Kernels::layout_bytes`]`(ty, WeightLayout::Device, n_out, n_in)` bytes, both
    /// 16-byte aligned on the stream's device and valid until the launch has run;
    /// the two must not overlap.
    pub unsafe fn rearrange(
        &self,
        stream: &Stream,
        ty: GgmlType,
        src: DevicePtr,
        dst: DevicePtr,
        n_out: u32,
        n_in: u32,
    ) -> Result<()> {
        let kernels = self.prefill.layouts.of(ty)?;
        let Some(bytes) = Self::layout_bytes(ty, WeightLayout::Device, n_out, n_in) else {
            return Err(Error::Invalid {
                what: "weight matrix",
                detail: format!(
                    "{n_in} columns of {ty} are not a whole number of {COLUMNS_MULTIPLE} values"
                ),
            });
        };
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", src, 16)?;
        check_alignment("rearranged matrix", dst, 16)?;
        // One thread per 32-bit word of the layout.
        let blocks = u32::try_from((bytes / 4).div_ceil(256)).map_err(|_| Error::Invalid {
            what: "device layout",
            detail: format!("{n_out} x {n_in} needs more blocks than a grid has"),
        })?;
        let args = Args::new().ptr(src).ptr(dst).u32(n_out).u32(n_in);
        // SAFETY: the argument list matches the ares_repack_* kernels; the ranges are
        // the caller's guarantee.
        unsafe { kernels.repack.launch(stream, [blocks, 1, 1], [256, 1, 1], 0, args) }
    }

    /// Which kernel serves which number of rows for matrices of `ty` in the device
    /// layout: the table [`Kernels::measure_layout`] measured, or the kept one.
    pub fn layout_table(&self, ty: GgmlType) -> Result<LayoutTable> {
        let prefill = &self.prefill;
        prefill.layouts.of(ty)?;
        let tables = prefill.layouts.tables.lock().unwrap_or_else(|e| e.into_inner());
        Ok(tables
            .iter()
            .find(|t| t.ty == ty)
            .cloned()
            .unwrap_or_else(|| LayoutTable::kept(ty, prefill.full_rate)))
    }

    /// Measures which kernel is fastest for each bucket of rows on this device and
    /// keeps the table for [`Kernels::product`]. A loader calls it once per type its
    /// model uses, before the weights are placed: it takes a matrix larger than the
    /// L2 cache with as many rows as the device computes at once (16 to 96 MiB of its
    /// memory, freed on return) and about 50 ms. Kernels within 5 % of the fastest of
    /// a bucket count as equal; of those [`Precision::Fast`] gets one with f32
    /// accumulation if there is one, and the narrowest; [`Precision::Exact`] the same
    /// among those with f32 accumulation.
    ///
    /// The device should be idle: a product of another context that runs at the same
    /// time is measured into the table.
    pub fn measure_layout(&self, ty: GgmlType) -> Result<LayoutTable> {
        let prefill = &self.prefill;
        let kernels = prefill.layouts.of(ty)?;
        let table = measure(&self.device, kernels)?;
        let mut tables = prefill.layouts.tables.lock().unwrap_or_else(|e| e.into_inner());
        tables.retain(|t| t.ty != ty);
        tables.push(table.clone());
        Ok(table)
    }

    /// The prepared form of the activations that [`Kernels::product`] reads for a
    /// matrix of `ty` in `layout` and `n_rows` rows.
    pub fn activation_form(
        &self,
        ty: GgmlType,
        layout: WeightLayout,
        precision: Precision,
        n_rows: u32,
    ) -> Result<ActivationForm> {
        match layout {
            WeightLayout::Device => {
                self.prefill.layouts.of(ty)?;
                Ok(ActivationForm::F16)
            }
            WeightLayout::File => Ok(match self.file_path(ty, precision, n_rows)? {
                None | Some(PrefillPath::Int8) => ActivationForm::Int8,
                Some(PrefillPath::F16 | PrefillPath::F16Flush) => ActivationForm::F16,
            }),
        }
    }

    /// The bytes of `n_rows` rows of `n_in` values in the prepared `form`.
    pub fn activation_bytes(form: ActivationForm, n_in: u32, n_rows: u32) -> u64 {
        match form {
            ActivationForm::Int8 => Self::quantized_activation_bytes(n_in, n_rows),
            ActivationForm::F16 => Self::prefill_activation_bytes(PrefillPath::F16, n_in, n_rows),
        }
    }

    /// Prepares `n_rows` rows of `n_in` f32 values at `x` in `form` at `out`.
    ///
    /// # Safety
    ///
    /// `x` must address the rows and `out`
    /// [`Kernels::activation_bytes`]`(form, n_in, n_rows)` bytes, 16-byte aligned, on
    /// the stream's device, valid until the launch has run. The products that read the
    /// f16 form are fastest when `out` starts on a line ([`Kernels::ACTIVATION_LINE`]).
    pub unsafe fn prepare_activations(
        &self,
        stream: &Stream,
        form: ActivationForm,
        x: DevicePtr,
        out: DevicePtr,
        n_in: u32,
        n_rows: u32,
    ) -> Result<()> {
        let path = match form {
            ActivationForm::Int8 if n_rows <= MAX_BATCH => {
                // SAFETY: the caller's guarantee is the one of the function it forwards to.
                return unsafe { self.quantize_activations(stream, x, out, n_in, n_rows) };
            }
            ActivationForm::Int8 => PrefillPath::Int8,
            ActivationForm::F16 => PrefillPath::F16,
        };
        // SAFETY: as above.
        unsafe { self.prepare_prefill_activations(stream, path, x, out, n_in, n_rows) }
    }

    /// Which kernel family multiplies a matrix of `ty` in the file's layout with
    /// `n_rows` rows: `None` for the decode kernels.
    fn file_path(
        &self,
        ty: GgmlType,
        precision: Precision,
        n_rows: u32,
    ) -> Result<Option<PrefillPath>> {
        let decode = Self::has_fast_matvec(ty) && n_rows <= MAX_BATCH;
        let decode_rows = Self::decode_kernel_rows(ty).unwrap_or(MAX_BATCH).min(MAX_BATCH);
        match precision {
            Precision::Exact => Ok(Some(PrefillPath::F16)),
            Precision::Fast if decode && n_rows <= decode_rows => Ok(None),
            Precision::Fast => Ok(Some(self.prefill_path_for(ty, Precision::Fast)?)),
        }
    }

    /// `y[b][r] = sum over c of w[r][c] * x[b][c]` for a matrix of `ty` that lies in
    /// `layout`, with any number of activation rows: the one product a model calls,
    /// whatever the type and the layout. `act` is the rows in the form
    /// [`Kernels::activation_form`] names for the same arguments.
    ///
    /// For a matrix in the device layout up to [`FEW_ROWS`] rows are the few-row kernels,
    /// launched as [`Kernels::few_plan`] says: the result of a row is the same bits whatever
    /// else the batch holds and whichever launch it is, so a step of one row and a verify
    /// window of drafts compute the same numbers for the row they share; it is the exact
    /// product of the f16 operands, summed in f32 (0.04 % rms from the exact f32 product,
    /// the rounding of the f16 operands). Above that the kernel comes from the table of the
    /// device ([`Kernels::layout_table`]): [`Precision::Fast`] takes f16 accumulation on a
    /// capped card (0.06 %). For one in the file's layout it is a decode kernel up to
    /// [`Kernels::decode_kernel_rows`] rows and a tile kernel above.
    ///
    /// # Safety
    ///
    /// `w` must address the matrix ([`Kernels::layout_bytes`]), `act` the prepared
    /// rows and `y` `n_rows * n_out` f32 values on the stream's device, aligned as
    /// [`Kernels::layout_alignment`] and the prepared form ask, and all valid until
    /// the launch has run.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn product(
        &self,
        stream: &Stream,
        ty: GgmlType,
        layout: WeightLayout,
        precision: Precision,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        match layout {
            WeightLayout::File => match self.file_path(ty, precision, n_rows)? {
                // SAFETY: the caller's guarantee is the one of the decode kernels.
                None => unsafe { self.matvec(stream, ty, w, act, y, n_in, n_out, n_rows) },
                // SAFETY: the caller's guarantee is the one of the tile kernels.
                Some(path) => unsafe {
                    self.matmul(stream, ty, path, w, act, y, n_in, n_out, n_rows)
                },
            },
            WeightLayout::Device if n_rows <= FEW_ROWS => {
                let plan = self.few_plan(ty, n_out, n_in)?;
                // SAFETY: the caller's guarantee.
                unsafe { self.few_launch(stream, ty, plan, w, act, y, n_in, n_out, n_rows) }
            }
            WeightLayout::Device => {
                let shape = self.layout_table(ty)?.shape(precision, n_rows);
                // SAFETY: the caller's guarantee.
                unsafe { self.layout_launch(stream, ty, shape, w, act, y, n_in, n_out, n_rows) }
            }
        }
    }

    /// Checks what both kinds of launch need and returns the kernels of the type.
    fn layout_checked(
        &self,
        ty: GgmlType,
        w: DevicePtr,
        act: DevicePtr,
        n_in: u32,
        n_rows: u32,
    ) -> Result<&TypeKernels> {
        let kernels = self.prefill.layouts.of(ty)?;
        if !Self::has_device_layout(ty, n_in) || n_rows == 0 || n_in.checked_mul(n_rows).is_none() {
            return Err(Error::Invalid {
                what: "layout kernel batch",
                detail: format!(
                    "{n_rows} rows of {n_in} values of {ty}: not whole steps, empty or too many"
                ),
            });
        }
        check_alignment("rearranged matrix", w, 16)?;
        check_alignment("prepared activations", act, 16)?;
        Ok(kernels)
    }

    /// Launches one kernel of the family over the device layout for 17 rows and more.
    #[allow(clippy::too_many_arguments)]
    unsafe fn layout_launch(
        &self,
        stream: &Stream,
        ty: GgmlType,
        shape: Shape,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        let kernels = self.layout_checked(ty, w, act, n_in, n_rows)?;
        if n_out == 0 {
            return Ok(());
        }
        let index = SHAPES.iter().position(|s| s.name == shape.name).expect("a shape of SHAPES");
        let n_weight_tiles = n_out.div_ceil(shape.weight_rows);
        let n_batch_tiles = n_rows.div_ceil(shape.batch_rows);
        let blocks =
            u32::try_from(u64::from(n_weight_tiles) * u64::from(n_batch_tiles)).map_err(|_| {
                Error::Invalid {
                    what: "layout kernel batch",
                    detail: format!("{n_out} x {n_rows} needs more blocks than a grid has"),
                }
            })?;
        let args =
            Args::new().ptr(w).ptr(act).ptr(y).u32(n_in).u32(n_out).u32(n_rows).u32(group_of(
                self.device.info().l2_cache_bytes,
                u64::from(shape.batch_rows) * u64::from(n_in) * 2,
                n_batch_tiles,
            ));
        // SAFETY: the argument list matches the ares_mmr_* kernels; the ranges are the
        // caller's guarantee.
        unsafe {
            kernels.shapes[index].launch(
                stream,
                [blocks, 1, 1],
                [shape.threads, 1, 1],
                shape.shared_bytes(),
                args,
            )
        }
    }

    /// Launches a few-row kernel (`n_rows` up to [`FEW_ROWS`]) as `plan` says.
    #[allow(clippy::too_many_arguments)]
    unsafe fn few_launch(
        &self,
        stream: &Stream,
        ty: GgmlType,
        plan: FewPlan,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        let kernels = self.layout_checked(ty, w, act, n_in, n_rows)?;
        if !plan.is_valid() || n_rows > FEW_ROWS {
            return Err(Error::Invalid {
                what: "few-row kernel",
                detail: format!("{plan:?} with {n_rows} rows: not a launch of the family"),
            });
        }
        if n_out == 0 {
            return Ok(());
        }
        let nb = n_rows.div_ceil(8);
        let tiles = n_out.div_ceil(16);
        let blocks = tiles.div_ceil(plan.block_warps >> plan.log2_warps_per_tile);
        let args = Args::new()
            .ptr(w)
            .ptr(act)
            .ptr(y)
            .u32(n_in)
            .u32(n_out)
            .u32(n_rows)
            .u32(plan.log2_warps_per_tile);
        // SAFETY: the argument list matches the ares_mmf_* kernels; the ranges are the
        // caller's guarantee.
        unsafe {
            kernels.few[usize::from(nb == 2)][plan.block()].launch(
                stream,
                [blocks, 1, 1],
                [plan.block_warps * 32, 1, 1],
                plan.shared_bytes(nb),
                args,
            )
        }
    }

    fn prefill_tuned_at(&self) -> Result<Arc<Mutex<Option<Instant>>>> {
        Ok(self.prefill.layouts.tuned_at.clone())
    }

    /// The launch a few-row product of a matrix of `n_out` rows of `n_in` values takes:
    /// the one [`Kernels::tune_layout`] measured for the shape, or [`FewPlan::untuned`].
    pub fn few_plan(&self, ty: GgmlType, n_out: u32, n_in: u32) -> Result<FewPlan> {
        let prefill = &self.prefill;
        let kernels = prefill.layouts.of(ty)?;
        let plans = prefill.layouts.plans.lock().unwrap_or_else(|e| e.into_inner());
        Ok(plans.get(&(ty, n_out, n_in)).copied().unwrap_or_else(|| {
            let device_warps =
                kernels.warps_per_multiprocessor * self.device.info().multiprocessors;
            FewPlan::untuned(n_out.div_ceil(16), device_warps)
        }))
    }

    /// Measures which launch ([`FewPlan::all`]) is the fastest for the few-row products of
    /// matrices of `ty` with `n_out` rows of `n_in` values on this device and keeps it for
    /// [`Kernels::product`]; it returns the plan, and the same without measuring again if it
    /// has been measured. A loader calls it once per shape of its model, before the weights
    /// are placed: it walks as many copies of a matrix of that shape as make 64 MiB or twice
    /// the L2 cache, at most 256 MiB (freed on return), in a few milliseconds. The first
    /// call on a device that has been idle also keeps it busy for 150 ms, for its clocks to
    /// rise.
    ///
    /// The plan is a matter of speed only: the product of a row is the same bits whichever
    /// launch computes it.
    ///
    /// The device should be idle: a product of another context that runs at the same
    /// time is measured into the plan.
    pub fn tune_layout(&self, ty: GgmlType, n_out: u32, n_in: u32) -> Result<FewPlan> {
        self.tuned(ty, n_out, n_in, None)
    }

    /// [`Kernels::tune_layout`] over a region of the device's memory that the caller
    /// lends instead of the copies it would allocate: the products walk through it, a
    /// launch after the other, from its start to its end. A caller that holds a region as
    /// large as its weights will be lends the translations of addresses the weights will
    /// need, and the plan is then the one of a step that reads each matrix from its own
    /// place; a matrix of a few megabytes measured over a few dozen finds them all, and
    /// the launches that wait on a translation, the small ones, look better than they are.
    /// What the region holds does not matter.
    pub fn tune_layout_over(
        &self,
        ty: GgmlType,
        n_out: u32,
        n_in: u32,
        region: &DeviceBuffer,
    ) -> Result<FewPlan> {
        self.tuned(ty, n_out, n_in, Some(region))
    }

    fn tuned(
        &self,
        ty: GgmlType,
        n_out: u32,
        n_in: u32,
        region: Option<&DeviceBuffer>,
    ) -> Result<FewPlan> {
        let prefill = &self.prefill;
        prefill.layouts.of(ty)?;
        let Some(bytes) = Self::layout_bytes(ty, WeightLayout::Device, n_out, n_in) else {
            return Err(Error::Invalid {
                what: "weight matrix",
                detail: format!(
                    "{n_in} columns of {ty} are not a whole number of {COLUMNS_MULTIPLE} values"
                ),
            });
        };
        {
            let plans = prefill.layouts.plans.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(plan) = plans.get(&(ty, n_out, n_in)) {
                return Ok(*plan);
            }
        }
        let plan = if n_out == 0 {
            FewPlan::ALL[0]
        } else {
            self.measure_few(ty, bytes, n_out, n_in, region)?
        };
        let mut plans = prefill.layouts.plans.lock().unwrap_or_else(|e| e.into_inner());
        plans.insert((ty, n_out, n_in), plan);
        Ok(plan)
    }

    fn measure_few(
        &self,
        ty: GgmlType,
        bytes: u64,
        n_out: u32,
        n_in: u32,
        region: Option<&DeviceBuffer>,
    ) -> Result<FewPlan> {
        /// Every candidate is timed this many times, in turn with the others, as a run
        /// of `BURST` launches; its fastest run counts.
        const ROUNDS: usize = 4;
        const BURST: u32 = 6;
        /// The copies of the matrix a launch walks through: together larger than the L2
        /// cache, so that every launch reads its weights from memory, as a step does.
        const MIN_BYTES: u64 = 64 << 20;
        const MAX_BYTES: u64 = 256 << 20;
        /// How long a device that was idle for longer than `RAMP_AFTER` is kept busy.
        const RAMP: Duration = Duration::from_millis(150);
        const RAMP_AFTER: Duration = Duration::from_millis(200);

        let device = &self.device;
        // The memory the launches walk: lent, or as many copies as it takes.
        let own;
        let (base, span) = match region {
            Some(region) if region.len() >= bytes => (region.ptr(), region.len()),
            _ => {
                let want =
                    (2 * u64::from(device.info().l2_cache_bytes)).clamp(MIN_BYTES, MAX_BYTES);
                let copies = want.div_ceil(bytes.max(1)).clamp(1, 64);
                own = DeviceBuffer::new(device, bytes * copies)?;
                // What the buffer holds does not matter to the time of a product (no kernel
                // branches on or addresses by a value) as long as it is numbers.
                let stream = Stream::new(device, 0)?;
                // SAFETY: the whole buffer of this device.
                unsafe { stream.fill(own.ptr(), 0x11, (bytes * copies) as usize)? };
                stream.synchronize()?;
                (own.ptr(), bytes * copies)
            }
        };
        let act_bytes = Kernels::activation_bytes(ActivationForm::F16, n_in, 1);
        let act = DeviceBuffer::new(device, act_bytes)?;
        let y = DeviceBuffer::new(device, u64::from(n_out) * 4)?;
        let stream = Stream::new(device, 0)?;
        // SAFETY: the whole buffer of this device.
        unsafe { stream.fill(act.ptr(), 0x3c, act_bytes as usize)? };
        let (start, end) = (Event::timed(device)?, Event::timed(device)?);
        // Where the next launch reads: behind the one before, from the start again at the end.
        let stride = bytes.next_multiple_of(256);
        let mut cursor = 0u64;
        let mut place = move || {
            if cursor + bytes > span {
                cursor = 0;
            }
            let at = base.offset(cursor);
            cursor += stride;
            at
        };
        // A device that has been idle runs at low clocks, which a few launches do not
        // raise: run for a while first.
        let idle = {
            let tuned_at = self.prefill_tuned_at()?;
            let mut tuned_at = tuned_at.lock().unwrap_or_else(|e| e.into_inner());
            let idle = tuned_at.is_none_or(|t| t.elapsed() > RAMP_AFTER);
            *tuned_at = Some(Instant::now());
            idle
        };
        let begun = Instant::now();
        while idle && begun.elapsed() < RAMP {
            for _ in 0..64 {
                // SAFETY: as below.
                unsafe {
                    self.few_launch(
                        &stream,
                        ty,
                        FewPlan::ALL[0],
                        place(),
                        act.ptr(),
                        y.ptr(),
                        n_in,
                        n_out,
                        1,
                    )?;
                }
            }
            stream.synchronize()?;
        }
        let mut best = [f32::INFINITY; FewPlan::ALL.len()];
        for round in 0..=ROUNDS {
            for (plan, best) in FewPlan::ALL.iter().zip(best.iter_mut()) {
                start.record(&stream)?;
                for _ in 0..BURST {
                    // SAFETY: the region holds a matrix of n_out x n_in at every place
                    // `place` names, `act` a row in the f16 form and `y` its products.
                    unsafe {
                        self.few_launch(
                            &stream,
                            ty,
                            *plan,
                            place(),
                            act.ptr(),
                            y.ptr(),
                            n_in,
                            n_out,
                            1,
                        )?;
                    }
                }
                end.record(&stream)?;
                end.synchronize()?;
                // The first round warms the caches up.
                if round > 0 {
                    *best = best.min(end.elapsed_ms_since(&start)? / BURST as f32);
                }
            }
        }
        // The fastest; of equals the first, which has the fewest warps on a tile.
        let fastest = best.iter().copied().fold(f32::INFINITY, f32::min);
        let pick = best.iter().position(|ms| *ms <= fastest).unwrap_or(0);
        Ok(FewPlan::ALL[pick])
    }

    /// The product with one launch of the few-row kernels, for benchmarks and tests that
    /// compare them.
    ///
    /// # Safety
    ///
    /// As [`Kernels::product`] with [`WeightLayout::Device`], f16 activations and at most
    /// [`FEW_ROWS`] rows.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn product_few(
        &self,
        stream: &Stream,
        ty: GgmlType,
        plan: FewPlan,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe { self.few_launch(stream, ty, plan, w, act, y, n_in, n_out, n_rows) }
    }

    /// The product with one named kernel of the family above [`FEW_ROWS`] rows, for
    /// benchmarks and tests that compare them: `kernel` is a [`LayoutKernel::name`]. (It
    /// takes any number of rows.)
    ///
    /// # Safety
    ///
    /// As [`Kernels::product`] with [`WeightLayout::Device`] and f16 activations.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn product_with(
        &self,
        stream: &Stream,
        ty: GgmlType,
        kernel: &str,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        let shape = SHAPES.iter().find(|s| s.name == kernel).ok_or_else(|| Error::Invalid {
            what: "layout kernel",
            detail: format!("no kernel {kernel} (there are {})", SHAPES.map(|s| s.name).join(", ")),
        })?;
        // SAFETY: the caller's guarantee.
        unsafe { self.layout_launch(stream, ty, *shape, w, act, y, n_in, n_out, n_rows) }
    }

    /// The names of the kernels of the family above [`FEW_ROWS`] rows
    /// ([`Kernels::product_with`]).
    pub fn layout_kernel_names() -> [&'static str; 4] {
        SHAPES.map(|s| s.name)
    }
}

/// Times the candidates of every bucket on a probe matrix and picks.
fn measure(device: &Device, kernels: &TypeKernels) -> Result<LayoutTable> {
    /// The values of a row of the probe matrix, within which they are chosen.
    const MIN_COLUMNS: u32 = 2048;
    const MAX_COLUMNS: u32 = 16384;
    /// The probe matrix is larger than L2, so that the kernels wait for memory as
    /// they do on a model's matrices, within these bounds.
    const MIN_BYTES: u64 = 16 << 20;
    const MAX_BYTES: u64 = 96 << 20;
    /// Every candidate is timed this many times, in turn with the others, as a run
    /// of `BURST` launches; its fastest run counts. Single launches with the host in
    /// between measure a card whose clocks have dropped, and whatever else ran then.
    const ROUNDS: usize = 4;
    const BURST: u32 = 4;

    let ty = kernels.ty;
    let device_info = device.info();
    // The rows: as many as the blocks of each kernel of up to 32 rows that the device
    // holds at once compute. With more, the blocks that do not fit at first run when
    // the others are done, each at the speed of one stream of loads, and the probe
    // would time that tail and not the kernel (a model's matrices meet it too, some
    // with one kernel and some with another; see `kernels/mmr.cuh`).
    let mut n_out = u32::MAX;
    for shape in [0, 1] {
        let held = kernels.shapes[shape]
            .blocks_per_multiprocessor(SHAPES[shape].threads, SHAPES[shape].shared_bytes())?;
        n_out = n_out.min(held * device_info.multiprocessors * SHAPES[shape].weight_rows);
    }
    let n_out = (n_out / 128 * 128).max(128);
    // The columns: enough that the matrix is larger than L2.
    let row_bytes_of = |n_in: u32| {
        Kernels::layout_bytes(ty, WeightLayout::Device, 16, n_in).expect("a layout") / 16
    };
    let want = (u64::from(device_info.l2_cache_bytes) * 5 / 4).clamp(MIN_BYTES, MAX_BYTES);
    let per_column =
        u64::from(n_out) * row_bytes_of(COLUMNS_MULTIPLE) / u64::from(COLUMNS_MULTIPLE);
    let n_in = u32::try_from(want / per_column.max(1))
        .unwrap_or(u32::MAX)
        .clamp(MIN_COLUMNS, MAX_COLUMNS)
        .next_multiple_of(COLUMNS_MULTIPLE);
    let row_bytes = row_bytes_of(n_in);
    let max_rows = PROBE_ROWS[PROBE_ROWS.len() - 1];
    let (w_bytes, act_bytes) = (
        u64::from(n_out) * row_bytes,
        Kernels::activation_bytes(ActivationForm::F16, n_in, max_rows),
    );
    let w = DeviceBuffer::new(device, w_bytes)?;
    let act = DeviceBuffer::new(device, act_bytes)?;
    let y = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_out) * 4)?;
    let stream = Stream::new(device, 0)?;
    // What the buffers hold does not matter to the time of a product (no kernel
    // branches on or addresses by a value) as long as it is numbers: small codes and
    // scales, activations near 1.
    // SAFETY: whole buffers of this device.
    unsafe {
        stream.fill(w.ptr(), 0x11, w_bytes as usize)?;
        stream.fill(act.ptr(), 0x3c, act_bytes as usize)?;
    }
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);

    let time = |shape: usize, n_rows: u32| -> Result<f32> {
        let info = SHAPES[shape];
        let n_batch_tiles = n_rows.div_ceil(info.batch_rows);
        let blocks = n_out.div_ceil(info.weight_rows) * n_batch_tiles;
        start.record(&stream)?;
        for _ in 0..BURST {
            let args = Args::new()
                .ptr(w.ptr())
                .ptr(act.ptr())
                .ptr(y.ptr())
                .u32(n_in)
                .u32(n_out)
                .u32(n_rows)
                .u32(group_of(
                    device_info.l2_cache_bytes,
                    u64::from(info.batch_rows) * u64::from(n_in) * 2,
                    n_batch_tiles,
                ));
            // SAFETY: the three buffers hold a matrix of n_out x n_in, max_rows rows of
            // activations and their products; the argument list matches the kernels.
            unsafe {
                kernels.shapes[shape].launch(
                    &stream,
                    [blocks, 1, 1],
                    [info.threads, 1, 1],
                    info.shared_bytes(),
                    args,
                )?;
            }
        }
        end.record(&stream)?;
        end.synchronize()?;
        Ok(end.elapsed_ms_since(&start)? / BURST as f32)
    };

    // The first pass over every candidate warms the clocks and the caches up.
    let mut times = CANDIDATES.map(|c| vec![f32::INFINITY; c.len()]);
    for round in 0..=ROUNDS {
        for (bucket, candidates) in CANDIDATES.iter().enumerate() {
            for (i, shape) in candidates.iter().enumerate() {
                let ms = time(*shape, PROBE_ROWS[bucket])?;
                if round > 0 {
                    times[bucket][i] = times[bucket][i].min(ms);
                }
            }
        }
    }

    let mut picks = [[0usize; 2]; 2];
    let mut ms = [[0.0f32; 2]; 2];
    for (bucket, candidates) in CANDIDATES.iter().enumerate() {
        let timed: Vec<(usize, f32)> =
            candidates.iter().copied().zip(times[bucket].iter().copied()).collect();
        // The candidates are listed narrowest first.
        let fastest = timed.iter().map(|(_, ms)| *ms).fold(f32::INFINITY, f32::min);
        let fastest_exact = timed
            .iter()
            .filter(|(shape, _)| SHAPES[*shape].exact)
            .map(|(_, ms)| *ms)
            .fold(f32::INFINITY, f32::min);
        let first = |exact_only: bool, limit: f32| {
            timed
                .iter()
                .copied()
                .find(|(shape, ms)| (SHAPES[*shape].exact || !exact_only) && *ms <= limit)
        };
        let exact = first(true, fastest_exact * AS_GOOD_WITHIN)
            .expect("every bucket has a kernel with f32 accumulation");
        let fast = first(true, fastest * AS_GOOD_WITHIN)
            .or_else(|| first(false, fastest * AS_GOOD_WITHIN))
            .expect("the fastest candidate is within the factor");
        picks[bucket] = [fast.0, exact.0];
        ms[bucket] = [fast.1, exact.1];
    }
    Ok(LayoutTable::new(ty, true, picks, ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every launch of the few-row kernels fits what its kernels are built for: the warps of
    /// a tile divide the block and the parts of K, and the parts the warps of a block hand
    /// to each other fit the shared memory a launch gets without asking.
    #[test]
    fn the_launches_of_the_few_row_kernels_fit_their_kernels() {
        for plan in FewPlan::all() {
            let warps = 1u32 << plan.log2_warps_per_tile;
            assert!(plan.is_valid());
            assert!(plan.block_warps.is_multiple_of(warps), "{plan:?}");
            assert!(FEW_PARTS.is_multiple_of(warps), "{plan:?}");
            for nb in [1, 2] {
                assert!(plan.shared_bytes(nb) <= DEFAULT_SHARED, "{plan:?} with {nb} fragments");
            }
            // One warp on a tile adds its parts in registers.
            assert_eq!(plan.shared_bytes(1) == 0, plan.log2_warps_per_tile == 0);
        }
        assert!(!FewPlan { block_warps: 4, log2_warps_per_tile: 3 }.is_valid());
        assert!(!FewPlan { block_warps: 6, log2_warps_per_tile: 0 }.is_valid());
    }

    /// The launch of a shape that was not measured puts more warps on a tile the fewer
    /// tiles the device has per warp it holds, and no more than it can hold.
    #[test]
    fn an_untuned_launch_uses_the_warps_of_the_device() {
        let mut last = u32::MAX;
        for tiles in [1, 20, 60, 100, 300, 1000, 5000] {
            let plan = FewPlan::untuned(tiles, 700);
            assert!(plan.is_valid());
            assert!(plan.log2_warps_per_tile <= last, "{tiles} tiles: {plan:?}");
            last = plan.log2_warps_per_tile;
            let warps = 1u32 << plan.log2_warps_per_tile;
            assert!(warps == 1 || tiles * warps <= 700, "{tiles} tiles: {plan:?}");
        }
        assert_eq!(FewPlan::untuned(1, 700).log2_warps_per_tile, 3);
        assert_eq!(FewPlan::untuned(5000, 700).log2_warps_per_tile, 0);
        assert!(FewPlan::untuned(0, 0).is_valid());
    }
}
