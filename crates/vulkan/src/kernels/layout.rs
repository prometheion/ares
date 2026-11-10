//! The device layouts of weight matrices: a matrix rearranged once at load, in the
//! bytes of the file's form, into the order the tensor cores read it
//! (`ares_quant::layout` is the specification; CUDA makes the same bytes), and the one
//! family of kernels (`shaders/mml_near.glsl`, `mml_wide.glsl`) that serves every
//! number of activation rows from that one copy. The counterpart of
//! `ares_cuda`'s `kernels/layout.rs`, with the same surface.
//!
//! A model holds one buffer per matrix and says in which [`Layout`] it lies. Which
//! prepared form of the activations a product reads, and which kernel computes it, is
//! this module's business ([`Kernels::activation_form`], [`Kernels::product`]): for a
//! matrix in the device layout a table measured on the device
//! ([`Kernels::measure_layout`]), for one in the file's layout the decode kernels up
//! to [`MAX_BATCH`] rows and the tile kernels above.
//!
//! The kernels over a device layout build the operands of the cooperative matrices in
//! registers, which works only where the device holds their elements in the order the
//! layout was made for; that is checked when the kernels are loaded, and a device that
//! fails the check (or has no cooperative matrices) keeps every matrix in the file's
//! layout ([`Kernels::layout_for`]).

use std::sync::{Arc, Mutex};

use ares_core::GgmlType;

use super::{Kernels, MAX_BATCH};
use crate::device::{ComponentType, Device};
use crate::error::{Error, Result};
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::module::{Args, Lazy, Module};
use crate::prefill::{Precision, PrefillPath, has_shape};
use crate::stream::{Event, Stream};
use crate::weights::{check_alignment, image};

/// How the bytes of a weight matrix lie on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// As in the model file. Every type has kernels for it: the decode kernels
    /// ([`Kernels::matvec`]) and the tile kernels ([`Kernels::matmul`]).
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
    /// The f32 rows as they are: what the decode kernels of the float types (F32, F16,
    /// BF16 matrices in the file's layout) read. CUDA's int8 form carries this copy
    /// inside; here it is a form of its own, so that no other product pays for it.
    F32,
}

/// One kernel of the family over a device layout: the weight rows and activation
/// rows a workgroup computes, and whether it accumulates in f32.
#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    weight_rows: u32,
    batch_rows: u32,
    exact: bool,
}

/// The kernels of a layout, `mml_<name>_<type>` (`LAYOUT_SHAPES` in `build.rs`),
/// ordered by activation rows. The first two read the activations and hold the
/// result in registers, and split the columns over the subgroups of a workgroup
/// (`mml_near.glsl`); the others stage the activations in shared memory
/// (`mml_wide.glsl`).
const SHAPES: [Shape; 6] = [
    Shape { name: "r8f", weight_rows: 32, batch_rows: 8, exact: true },
    Shape { name: "r16f", weight_rows: 32, batch_rows: 16, exact: true },
    Shape { name: "r32f", weight_rows: 128, batch_rows: 32, exact: true },
    Shape { name: "r32h", weight_rows: 128, batch_rows: 32, exact: false },
    Shape { name: "r64f", weight_rows: 128, batch_rows: 64, exact: true },
    Shape { name: "r64h", weight_rows: 128, batch_rows: 64, exact: false },
];
/// The shapes of `mml_near.glsl`.
const NEAR_SHAPES: usize = 2;

/// The numbers of activation rows a table distinguishes: up to 8, 16 and 32 rows,
/// and more.
const BUCKETS: [u32; 4] = [8, 16, 32, u32::MAX];
/// The rows each bucket is measured with. The last is two activation tiles of the
/// widest shape, so that it runs grouped as a prompt does.
const PROBE_ROWS: [u32; 4] = [8, 16, 32, 128];
/// The shapes measured for each bucket. Up to 16 rows only the two that split the
/// columns over the subgroups of a workgroup: the wider ones run one chain over the
/// columns per workgroup, which bounds them on a matrix with few rows (2048 x 17408:
/// 0.14 ms against 0.05 on an RTX 4070 at any number of rows up to 32), and the probe
/// matrix, which is large, would not show it. They gain at most 5 % on a large one.
const CANDIDATES: [&[usize]; 4] = [&[0, 1], &[0, 1], &[2, 3, 4, 5], &[4, 5]];
/// A kernel with f32 accumulation is taken for [`Precision::Fast`] when it is within
/// this factor of the fastest: exact arithmetic where it costs nothing.
const EXACT_WITHIN: f32 = 1.03;
/// The most activation tiles that take a weight tile together (`group` of the
/// kernels). 4, 8, 12 and 16 measured within 2 % of each other at 1024 rows on an RTX
/// 4070 and an RTX 3060 (16 lost 5 % on the second).
const GROUP_MAX: u32 = 8;

/// The types with a device layout and their kernels' name part. All of them step
/// through a row in super-blocks of 256 values, so a row must be a whole number of
/// those.
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
    /// The shape's name (`r8f`, `r16f`, `r32f`, `r32h`, `r64f`, `r64h`).
    pub name: &'static str,
    /// The weight rows and the activation rows one workgroup computes.
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

/// Which kernel serves which number of activation rows for one type on one device.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutTable {
    pub ty: GgmlType,
    /// Whether the table was measured on this device ([`Kernels::measure_layout`]).
    /// Otherwise it is the one that follows from what the device reports: the
    /// narrowest shape for each bucket, with f32 accumulation up to 16 rows, and
    /// above that too where the tensor cores run it at full rate.
    pub measured: bool,
    pub rows: [LayoutRows; 4],
    /// The time of the same probe products with the kernels of the file's layout
    /// (the decode kernel at 8 rows, in two launches at 16, the tile kernel above),
    /// in milliseconds, for comparing the two layouts on this device; 0 where it was
    /// not measured.
    pub file_ms: [f32; 4],
    /// The shape of each bucket, `[fast, exact]`.
    picks: [[usize; 2]; 4],
}

impl LayoutTable {
    fn new(ty: GgmlType, measured: bool, picks: [[usize; 2]; 4], ms: [[f32; 2]; 4]) -> Self {
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
        Self { ty, measured, rows, file_ms: [0.0; 4], picks }
    }

    /// The table of a device that was not measured.
    fn kept(ty: GgmlType, full_rate: bool, near: bool) -> Self {
        let wide = if full_rate { [2, 4] } else { [3, 5] };
        let few = if near { [[0, 0], [1, 1]] } else { [[wide[0], 2], [wide[0], 2]] };
        Self::new(ty, false, [few[0], few[1], [wide[0], 2], [wide[1], 4]], [[0.0; 2]; 4])
    }

    fn shape(&self, precision: Precision, n_rows: u32) -> Shape {
        let bucket = BUCKETS.iter().position(|top| n_rows <= *top).unwrap_or(BUCKETS.len() - 1);
        SHAPES[self.picks[bucket][usize::from(precision == Precision::Exact)]]
    }
}

/// The kernels of one type with a device layout.
#[derive(Debug, Clone)]
struct TypeKernels {
    ty: GgmlType,
    repack: Arc<Lazy>,
    shapes: [Arc<Lazy>; 6],
}

/// The kernels over the device layouts on one device, and the tables measured so
/// far.
#[derive(Debug, Clone)]
pub(super) struct Layouts {
    types: Vec<TypeKernels>,
    tables: Arc<Mutex<Vec<LayoutTable>>>,
    /// Whether the kernels of few rows can run: the device has f16 products of 16 x 16
    /// by 16 x 8 into f32 and holds the elements of those operands in the order the
    /// kernels read and write them (`coop_layout_check8.comp`).
    near: bool,
    /// Whether f16 products run as fast with f32 accumulation as with f16 accumulation.
    full_rate: bool,
}

impl Layouts {
    /// The kernels, on a device whose cooperative matrices of use A can be built in
    /// registers (`register_operand`, checked by the prefill kernels' loader).
    pub(super) fn load(device: &Device, full_rate: bool) -> Result<Self> {
        let types = TYPES
            .iter()
            .map(|(ty, suffix)| {
                let shapes = SHAPES
                    .iter()
                    .map(|shape| {
                        Ok(Lazy::new(device, image(&format!("mml_{}_{suffix}", shape.name))?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(TypeKernels {
                    ty: *ty,
                    repack: Lazy::new(device, image(&format!("mml_repack_{suffix}"))?),
                    shapes: <[Arc<Lazy>; 6]>::try_from(shapes).expect("six shapes"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let near = has_shape(
            device,
            (16, 8, 16),
            ComponentType::F16,
            ComponentType::F16,
            ComponentType::F32,
        ) && check_near_operands(device).unwrap_or(false);
        Ok(Self { types, tables: Arc::default(), near, full_rate })
    }

    /// Makes the kernels of `types` now, on all cores.
    pub(super) fn prepare(&self, types: &[GgmlType]) -> Result<()> {
        let all: Vec<&Arc<Lazy>> = self
            .types
            .iter()
            .filter(|t| types.contains(&t.ty))
            .flat_map(|t| std::iter::once(&t.repack).chain(self.usable(t)))
            .collect();
        super::parallel_map(&all, |lazy| lazy.get().map(|_| ())).map(|_| ())
    }

    /// The product kernels of a type that this device can run.
    fn usable<'a>(&self, kernels: &'a TypeKernels) -> impl Iterator<Item = &'a Arc<Lazy>> {
        kernels.shapes.iter().skip(if self.near { 0 } else { NEAR_SHAPES })
    }

    fn of(&self, ty: GgmlType) -> Result<&TypeKernels> {
        self.types.iter().find(|t| t.ty == ty).ok_or_else(|| no_layout(ty))
    }

    /// The candidates of a bucket on this device.
    fn candidates(&self, bucket: usize) -> Vec<usize> {
        let all: Vec<usize> =
            CANDIDATES[bucket].iter().copied().filter(|s| self.near || *s >= NEAR_SHAPES).collect();
        // Without the kernels of few rows the narrowest wide ones serve them.
        if all.is_empty() { vec![2, 3] } else { all }
    }
}

fn no_layout(ty: GgmlType) -> Error {
    Error::invalid(
        "device layout",
        format!(
            "{ty} has none (the types that have one: {})",
            TYPES.map(|(t, _)| t.name()).join(", ")
        ),
    )
}

/// Whether the device holds the elements of a B operand and of an accumulator of
/// 16 x 8 in the order `mml_near.glsl` reads and writes them: `coop_layout_check8.comp`
/// builds one by that rule and reads the other, and both must come out as the matrix
/// that was meant.
fn check_near_operands(device: &Device) -> Result<bool> {
    let stream = Stream::new(device)?;
    let out = DeviceBuffer::new(device, 256 * 4)?;
    let function = Module::load(device, image("coop_layout_check8")?)?.function("main")?;
    // SAFETY: `out` holds the 256 floats the kernel writes.
    unsafe { function.launch(&stream, [1, 1, 1], Args::new().ptr(out.ptr()))? };
    stream.synchronize()?;
    let mut bytes = [0u8; 256 * 4];
    out.read(0, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .all(|(i, b)| f32::from_le_bytes(*b) == (i % 128 + 1) as f32))
}

/// The group of activation tiles that take every weight tile in turn: as many as
/// [`GROUP_MAX`], and a divisor of the tiles when one is near, so that every wave of
/// workgroups works on whole groups.
fn group_of(n_batch_tiles: u32) -> u32 {
    let group = GROUP_MAX.min(n_batch_tiles);
    (1..=group)
        .rev()
        .find(|g| n_batch_tiles.is_multiple_of(*g))
        .filter(|g| *g * 2 > group)
        .unwrap_or(group)
}

impl Kernels {
    fn layouts(&self) -> Result<&Layouts> {
        self.0.layouts.as_ref().ok_or_else(|| Error::Unsupported {
            device: self.0.device.info().name.clone(),
            reason: "the kernels over the device layouts need f16 cooperative matrices whose \
                     operands can be built in registers"
                .into(),
        })
    }

    /// Whether a matrix of `ty` with rows of `n_in` values has a device layout
    /// ([`Layout::Device`]): the type has one and the rows are whole steps of it. On a
    /// device without the kernels for it nothing reads it; [`Kernels::layout_for`]
    /// asks both.
    pub fn has_device_layout(ty: GgmlType, n_in: u32) -> bool {
        TYPES.iter().any(|(t, _)| *t == ty) && n_in != 0 && n_in.is_multiple_of(COLUMNS_MULTIPLE)
    }

    /// Whether this device has the kernels over the device layouts at all: f16
    /// cooperative matrices of 16 x 16 with f16 and f32 accumulation, a subgroup of 32,
    /// and operands whose elements lie in the order the layouts were made for (checked
    /// when the kernels were loaded).
    pub fn has_layout_kernels(&self) -> bool {
        self.0.layouts.is_some()
    }

    /// The layout a loader should give a matrix of `ty` with rows of `n_in` values on
    /// this device: the device layout where the type has one and the device the
    /// kernels for it, the file's otherwise.
    pub fn layout_for(&self, ty: GgmlType, n_in: u32) -> Layout {
        if self.0.layouts.is_some() && Self::has_device_layout(ty, n_in) {
            Layout::Device
        } else {
            Layout::File
        }
    }

    /// The bytes of a matrix of `n_out` rows of `n_in` values in `layout`; `None` when
    /// the type has no such layout or the rows are not whole blocks. The device
    /// layout takes the bytes of the file's with the rows rounded up to a multiple
    /// of 16.
    pub fn layout_bytes(ty: GgmlType, layout: Layout, n_out: u32, n_in: u32) -> Option<u64> {
        let (values, bytes) = ty.block();
        if !n_in.is_multiple_of(values) {
            return None;
        }
        let row = u64::from(n_in / values) * u64::from(bytes);
        match layout {
            Layout::File => Some(u64::from(n_out) * row),
            Layout::Device => Self::has_device_layout(ty, n_in)
                .then(|| u64::from(n_out.next_multiple_of(16)) * row),
        }
    }

    /// The alignment, in bytes, a matrix needs on the device: 16 in either layout,
    /// which is the widest load of any kernel that may read it.
    pub fn layout_alignment(_ty: GgmlType, _layout: Layout) -> u64 {
        16
    }

    /// Rearranges the matrix at `src` (`n_out` rows of `n_in` values of `ty` in the
    /// file's layout) into its device layout at `dst`, the bytes
    /// `ares_quant::layout::rearrange` makes. One kernel; it runs once per matrix, at
    /// load.
    ///
    /// A matrix that a tensor split cuts into rows or columns is cut first, in the
    /// file's layout, and each part rearranged on the device that holds it: the
    /// layout mixes 16 rows, so a part cannot be cut out of a rearranged matrix
    /// except at multiples of 16 rows and 256 columns.
    ///
    /// # Safety
    ///
    /// `src` must address the matrix and `dst`
    /// [`Kernels::layout_bytes`]`(ty, Layout::Device, n_out, n_in)` bytes, both
    /// 16-byte aligned (checked) on the stream's device and valid until the launch
    /// has run; the two must not overlap.
    pub unsafe fn rearrange(
        &self,
        stream: &Stream,
        ty: GgmlType,
        src: DevicePtr,
        dst: DevicePtr,
        n_out: u32,
        n_in: u32,
    ) -> Result<()> {
        let kernels = self.layouts()?.of(ty)?;
        let Some(bytes) = Self::layout_bytes(ty, Layout::Device, n_out, n_in) else {
            return Err(Error::invalid(
                "weight matrix",
                format!(
                    "{n_in} columns of {ty} are not a whole number of {COLUMNS_MULTIPLE} values"
                ),
            ));
        };
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("weight matrix", src, 16)?;
        check_alignment("rearranged matrix", dst, 16)?;
        // One invocation per 32-bit word of the layout.
        let groups = u32::try_from((bytes / 4).div_ceil(256)).map_err(|_| {
            Error::invalid("device layout", format!("{n_out} x {n_in} needs too many workgroups"))
        })?;
        let args = Args::new().ptr(src).ptr(dst).u32(n_out).u32(n_in);
        // SAFETY: the argument list matches mml_repack.glsl; the ranges are the caller's
        // guarantee.
        unsafe {
            kernels.repack.get()?.launch(stream, super::grid_1d(groups, self.grid_limit()), args)
        }
    }

    /// Which kernel serves which number of rows for matrices of `ty` in the device
    /// layout: the table [`Kernels::measure_layout`] measured, or the one that
    /// follows from what the device reports.
    pub fn layout_table(&self, ty: GgmlType) -> Result<LayoutTable> {
        let layouts = self.layouts()?;
        layouts.of(ty)?;
        let tables = layouts.tables.lock().unwrap_or_else(|e| e.into_inner());
        Ok(tables
            .iter()
            .find(|t| t.ty == ty)
            .cloned()
            .unwrap_or_else(|| LayoutTable::kept(ty, layouts.full_rate, layouts.near)))
    }

    /// Measures which kernel is fastest for each bucket of rows on this device and
    /// keeps the table for [`Kernels::product`]. A loader calls it once per type its
    /// model uses, before the weights are placed: it takes a probe matrix of about
    /// 48 MiB (larger than the caches of the cards it was written on; freed on
    /// return) and about 100 ms. [`Precision::Fast`] gets the fastest kernel of a
    /// bucket, or the one with f32 accumulation when that is within 3 % of it;
    /// [`Precision::Exact`] the fastest with f32 accumulation. The same products are
    /// timed with the kernels of the file's layout ([`LayoutTable::file_ms`]), so that
    /// a loader can see what the device layout gains or costs here.
    ///
    /// The device should be idle: a product of another context that runs at the same
    /// time is measured into the table.
    pub fn measure_layout(&self, ty: GgmlType) -> Result<LayoutTable> {
        let layouts = self.layouts()?;
        let kernels = layouts.of(ty)?;
        let table = self.measure(layouts, kernels)?;
        let mut tables = layouts.tables.lock().unwrap_or_else(|e| e.into_inner());
        tables.retain(|t| t.ty != ty);
        tables.push(table.clone());
        Ok(table)
    }

    /// The prepared form of the activations that [`Kernels::product`] reads for a
    /// matrix of `ty` in `layout` and `n_rows` rows.
    pub fn activation_form(
        &self,
        ty: GgmlType,
        layout: Layout,
        precision: Precision,
        n_rows: u32,
    ) -> Result<ActivationForm> {
        match layout {
            Layout::Device => {
                self.layouts()?.of(ty)?;
                Ok(ActivationForm::F16)
            }
            Layout::File => Ok(match self.file_path(ty, precision, n_rows)? {
                None if Self::needs_quantized_activations(ty) => ActivationForm::Int8,
                None => ActivationForm::F32,
                Some(PrefillPath::Int8) => ActivationForm::Int8,
                Some(PrefillPath::F16 | PrefillPath::F16Flush) => ActivationForm::F16,
            }),
        }
    }

    /// The bytes of `n_rows` rows of `n_in` values in the prepared `form`.
    pub fn activation_bytes(form: ActivationForm, n_in: u32, n_rows: u32) -> u64 {
        match form {
            ActivationForm::Int8 => Self::quantized_activation_bytes(n_in, n_rows),
            ActivationForm::F16 => Self::prefill_activation_bytes(PrefillPath::F16, n_in, n_rows),
            ActivationForm::F32 => u64::from(n_in) * u64::from(n_rows) * 4,
        }
    }

    /// Prepares `n_rows` rows of `n_in` f32 values at `x` in `form` at `out`. For
    /// [`ActivationForm::F32`] that is a copy; a caller that keeps `x` can pass it to
    /// the product instead.
    ///
    /// # Safety
    ///
    /// `x` must address the rows and `out`
    /// [`Kernels::activation_bytes`]`(form, n_in, n_rows)` bytes, 16-byte aligned, on
    /// the stream's device, valid until the launch has run and not overlapping.
    pub unsafe fn prepare_activations(
        &self,
        stream: &Stream,
        form: ActivationForm,
        x: DevicePtr,
        out: DevicePtr,
        n_in: u32,
        n_rows: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee is the one of the functions it forwards to.
        unsafe {
            match form {
                ActivationForm::Int8 => self.quantize_activation_rows(stream, x, out, n_in, n_rows),
                ActivationForm::F16 => {
                    self.prepare_prefill_activations(stream, PrefillPath::F16, x, out, n_in, n_rows)
                }
                ActivationForm::F32 => {
                    let bytes = Self::activation_bytes(form, n_in, n_rows);
                    let len = usize::try_from(bytes)
                        .map_err(|_| Error::invalid("activations", "size overflows usize"))?;
                    stream.copy(out, x, len)
                }
            }
        }
    }

    /// Which kernel family multiplies a matrix of `ty` in the file's layout with
    /// `n_rows` rows: `None` for the decode kernels.
    fn file_path(
        &self,
        ty: GgmlType,
        precision: Precision,
        n_rows: u32,
    ) -> Result<Option<PrefillPath>> {
        let tile = self.has_matmul(ty, PrefillPath::F16);
        let decode = Self::has_fast_matvec(ty) && n_rows <= MAX_BATCH;
        match precision {
            Precision::Exact if tile => Ok(Some(PrefillPath::F16)),
            Precision::Fast if decode => Ok(None),
            Precision::Fast if tile => Ok(Some(self.prefill_path(Precision::Fast)?)),
            // No tile kernel: the decode kernels are all there is.
            _ if decode => Ok(None),
            _ => Err(Error::invalid(
                "weight kernel",
                format!(
                    "{ty} in the file's layout has no kernel for {n_rows} rows in one launch on \
                     {} (Kernels::matmul_decode runs them in batches of {MAX_BATCH})",
                    self.0.device.info().name
                ),
            )),
        }
    }

    /// `y[b][r] = sum over c of w[r][c] * x[b][c]` for a matrix of `ty` that lies in
    /// `layout`, with any number of activation rows: the one product a model calls,
    /// whatever the type and the layout. `act` is the rows in the form
    /// [`Kernels::activation_form`] names for the same arguments.
    ///
    /// For a matrix in the device layout the kernel comes from the table of the
    /// device ([`Kernels::layout_table`]): up to 16 rows it accumulates in f32 (the
    /// rounding of the f16 operands is all its error, about 0.04 % rms of a sum's
    /// size), above that [`Precision::Fast`] takes f16 accumulation on a device that
    /// runs it faster (0.06 %). For one in the file's layout it is a decode kernel up
    /// to [`MAX_BATCH`] rows (integer products with 8-bit activations, about 1 %; with
    /// [`Precision::Exact`] the tile kernel instead) and a tile kernel above.
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
        layout: Layout,
        precision: Precision,
        w: DevicePtr,
        act: DevicePtr,
        y: DevicePtr,
        n_in: u32,
        n_out: u32,
        n_rows: u32,
    ) -> Result<()> {
        match layout {
            Layout::File => match self.file_path(ty, precision, n_rows)? {
                // SAFETY: the caller's guarantee is the one of the function it forwards to.
                None => unsafe { self.matvec(stream, ty, w, act, y, n_in, n_out, n_rows) },
                // SAFETY: as above.
                Some(path) => unsafe {
                    self.matmul(stream, ty, path, w, act, y, n_in, n_out, n_rows)
                },
            },
            Layout::Device => {
                let shape = self.layout_table(ty)?.shape(precision, n_rows);
                // SAFETY: the caller's guarantee.
                unsafe { self.layout_launch(stream, ty, shape, w, act, y, n_in, n_out, n_rows) }
            }
        }
    }

    /// Launches one kernel of the family over the device layout.
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
        let layouts = self.layouts()?;
        let kernels = layouts.of(ty)?;
        if !Self::has_device_layout(ty, n_in) || n_rows == 0 || n_in.checked_mul(n_rows).is_none() {
            return Err(Error::invalid(
                "layout kernel batch",
                format!(
                    "{n_rows} rows of {n_in} values of {ty}: not whole steps, empty or too many"
                ),
            ));
        }
        let index = SHAPES.iter().position(|s| s.name == shape.name).expect("a shape of SHAPES");
        if index < NEAR_SHAPES && !layouts.near {
            return Err(Error::Unsupported {
                device: self.0.device.info().name.clone(),
                reason: format!(
                    "the kernel {} needs cooperative matrices of 16 x 8 in the order of the layout",
                    shape.name
                ),
            });
        }
        if n_out == 0 {
            return Ok(());
        }
        check_alignment("rearranged matrix", w, 16)?;
        check_alignment("prepared activations", act, 16)?;
        let n_weight_tiles = n_out.div_ceil(shape.weight_rows);
        let n_batch_tiles = n_rows.div_ceil(shape.batch_rows);
        let groups =
            u32::try_from(u64::from(n_weight_tiles) * u64::from(n_batch_tiles)).map_err(|_| {
                Error::invalid(
                    "layout kernel batch",
                    format!("{n_out} x {n_rows} needs too many workgroups"),
                )
            })?;
        let args = Args::new()
            .ptr(w)
            .ptr(act)
            .ptr(y)
            .u32(n_in)
            .u32(n_out)
            .u32(n_rows)
            .u32(group_of(n_batch_tiles));
        // SAFETY: the argument list matches mml_near.glsl and mml_wide.glsl; the ranges
        // are the caller's guarantee.
        unsafe {
            kernels.shapes[index].get()?.launch(
                stream,
                super::grid_1d(groups, self.grid_limit()),
                args,
            )
        }
    }

    /// The product with one named kernel of the family, for benchmarks and tests that
    /// compare them: `kernel` is a [`LayoutKernel::name`].
    ///
    /// # Safety
    ///
    /// As [`Kernels::product`] with [`Layout::Device`] and f16 activations.
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
        let shape = SHAPES.iter().find(|s| s.name == kernel).ok_or_else(|| {
            Error::invalid(
                "layout kernel",
                format!("no kernel {kernel} (there are {})", SHAPES.map(|s| s.name).join(", ")),
            )
        })?;
        // SAFETY: the caller's guarantee.
        unsafe { self.layout_launch(stream, ty, *shape, w, act, y, n_in, n_out, n_rows) }
    }

    /// The names of the kernels of the family ([`Kernels::product_with`]).
    pub fn layout_kernel_names() -> [&'static str; 6] {
        SHAPES.map(|s| s.name)
    }

    /// Times the candidates of every bucket on a probe matrix and picks.
    fn measure(&self, layouts: &Layouts, kernels: &TypeKernels) -> Result<LayoutTable> {
        /// Rows of the probe matrix: 4096 values, whole steps of every layout.
        const N_IN: u32 = 4096;
        /// The probe matrix is larger than the caches, so that the kernels wait for
        /// memory as they do on a model's matrices; a device that has little memory
        /// free gets a smaller one.
        const PROBE_BYTES: u64 = 48 << 20;
        const MIN_BYTES: u64 = 4 << 20;
        /// Every candidate is timed this many times, in turn with the others, as a run
        /// of `BURST` launches; its fastest run counts. Single launches with the host
        /// in between measure a card whose clocks have dropped, and whatever else ran
        /// then.
        const ROUNDS: usize = 4;
        const BURST: u32 = 4;

        let device = &self.0.device;
        let ty = kernels.ty;
        let row_bytes = Self::layout_bytes(ty, Layout::Device, 16, N_IN).expect("a layout") / 16;
        let free = device.mem_info().map_or(PROBE_BYTES * 8, |m| m.free);
        let want = PROBE_BYTES.min(free / 8).max(MIN_BYTES);
        let n_out = u32::try_from(want / row_bytes).unwrap_or(u32::MAX).next_multiple_of(128);
        let max_rows = PROBE_ROWS[PROBE_ROWS.len() - 1];
        let w_bytes = u64::from(n_out) * row_bytes;
        let act_bytes = Self::activation_bytes(ActivationForm::F16, N_IN, max_rows)
            .max(Self::activation_bytes(ActivationForm::Int8, N_IN, max_rows))
            .next_multiple_of(4);
        let w = DeviceBuffer::new(device, w_bytes)?;
        let act = DeviceBuffer::new(device, act_bytes)?;
        let y = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_out) * 4)?;
        let stream = Stream::new(device)?;
        // What the buffers hold does not matter to the time of a product (no kernel
        // branches on or addresses by a value) as long as it is numbers: small codes
        // and scales, activations near 1.
        // SAFETY: whole buffers of this device.
        unsafe {
            stream.fill(w.ptr(), 0x11, w_bytes as usize)?;
            stream.fill(act.ptr(), 0x3c, act_bytes as usize)?;
        }
        let (start, end) = (Event::timed(device)?, Event::timed(device)?);
        let timed = |launch: &dyn Fn() -> Result<()>| -> Result<f32> {
            start.record(&stream)?;
            for _ in 0..BURST {
                launch()?;
            }
            end.record(&stream)?;
            end.synchronize()?;
            Ok(end.elapsed_ms_since(&start)? / BURST as f32)
        };
        let time = |shape: usize, n_rows: u32| -> Result<f32> {
            // SAFETY: the three buffers hold a matrix of n_out x N_IN, max_rows rows of
            // activations and their products.
            timed(&|| unsafe {
                self.layout_launch(
                    &stream,
                    ty,
                    SHAPES[shape],
                    w.ptr(),
                    act.ptr(),
                    y.ptr(),
                    N_IN,
                    n_out,
                    n_rows,
                )
            })
        };
        // The same products with the kernels of the file's layout, on the same bytes
        // (they are as valid a matrix in that layout as in the other).
        let time_file = |n_rows: u32| -> Result<f32> {
            timed(&|| {
                let mut done = 0;
                while done < n_rows {
                    let n =
                        if n_rows <= 2 * MAX_BATCH { MAX_BATCH.min(n_rows - done) } else { n_rows };
                    let y_at = y.ptr().offset(u64::from(done) * u64::from(n_out) * 4);
                    // SAFETY: as above; the activations are read as whatever form the
                    // kernel takes.
                    unsafe {
                        self.product(
                            &stream,
                            ty,
                            Layout::File,
                            Precision::Fast,
                            w.ptr(),
                            act.ptr(),
                            y_at,
                            N_IN,
                            n_out,
                            n,
                        )?;
                    }
                    done += n;
                }
                Ok(())
            })
        };

        // The first pass over every candidate warms the clocks and the caches up.
        let candidates: [Vec<usize>; 4] = std::array::from_fn(|b| layouts.candidates(b));
        let mut times = candidates.clone().map(|c| vec![f32::INFINITY; c.len()]);
        let mut file_ms = [f32::INFINITY; 4];
        for round in 0..=ROUNDS {
            for (bucket, shapes) in candidates.iter().enumerate() {
                for (i, shape) in shapes.iter().enumerate() {
                    let ms = time(*shape, PROBE_ROWS[bucket])?;
                    if round > 0 {
                        times[bucket][i] = times[bucket][i].min(ms);
                    }
                }
                let ms = time_file(PROBE_ROWS[bucket])?;
                if round > 0 {
                    file_ms[bucket] = file_ms[bucket].min(ms);
                }
            }
        }

        let mut picks = [[0usize; 2]; 4];
        let mut ms = [[0.0f32; 2]; 4];
        for (bucket, shapes) in candidates.iter().enumerate() {
            let timed: Vec<(usize, f32)> =
                shapes.iter().copied().zip(times[bucket].iter().copied()).collect();
            let by_time = |a: &&(usize, f32), b: &&(usize, f32)| a.1.total_cmp(&b.1);
            let fastest = *timed.iter().min_by(by_time).expect("candidates");
            let exact = *timed
                .iter()
                .filter(|(shape, _)| SHAPES[*shape].exact)
                .min_by(by_time)
                .expect("every bucket has a kernel with f32 accumulation");
            let fast = if exact.1 <= fastest.1 * EXACT_WITHIN { exact } else { fastest };
            picks[bucket] = [fast.0, exact.0];
            ms[bucket] = [fast.1, exact.1];
        }
        let mut table = LayoutTable::new(ty, true, picks, ms);
        table.file_ms = file_ms;
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_types_are_those_of_the_reference() {
        for (ty, _) in TYPES {
            assert!(Kernels::has_device_layout(ty, 512));
            assert!(!Kernels::has_device_layout(ty, 96));
            assert_eq!(
                Kernels::layout_bytes(ty, Layout::Device, 17, 512),
                Kernels::layout_bytes(ty, Layout::File, 32, 512)
            );
        }
        assert!(!Kernels::has_device_layout(GgmlType::IQ3_S, 512));
        assert_eq!(Kernels::layout_bytes(GgmlType::IQ3_S, Layout::Device, 16, 256), None);
        assert_eq!(Kernels::layout_bytes(GgmlType::Q4_K, Layout::File, 2, 256), Some(288));
    }

    #[test]
    fn a_table_without_a_measurement_takes_the_narrowest_shape() {
        let capped = LayoutTable::kept(GgmlType::Q4_K, false, true);
        assert_eq!(capped.shape(Precision::Fast, 1).name, "r8f");
        assert_eq!(capped.shape(Precision::Fast, 9).name, "r16f");
        assert_eq!(capped.shape(Precision::Fast, 17).name, "r32h");
        assert_eq!(capped.shape(Precision::Exact, 17).name, "r32f");
        assert_eq!(capped.shape(Precision::Fast, 5000).name, "r64h");
        assert_eq!(capped.shape(Precision::Exact, 33).name, "r64f");
        let full = LayoutTable::kept(GgmlType::Q4_K, true, true);
        assert_eq!(full.shape(Precision::Fast, 5000).name, "r64f");
        // Without the kernels of few rows the wide ones serve every bucket.
        let wide_only = LayoutTable::kept(GgmlType::Q4_K, false, false);
        assert_eq!(wide_only.shape(Precision::Fast, 1).name, "r32h");
        assert_eq!(wide_only.shape(Precision::Exact, 12).name, "r32f");
    }

    #[test]
    fn groups_divide_the_tiles_when_they_can() {
        assert_eq!(group_of(1), 1);
        assert_eq!(group_of(8), 8);
        assert_eq!(group_of(16), 8);
        assert_eq!(group_of(12), 6);
        assert_eq!(group_of(13), 8);
    }
}
