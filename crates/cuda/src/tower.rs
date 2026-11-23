//! The kernels of the image tower (`qwen3vl_merger`): what its pass needs
//! besides the matrix products, which are the tile kernels of [`Kernels`]
//! (`crate::Kernels::matmul` over activations in the f16 form of
//! `crate::Kernels::prepare_prefill_activations`).
//!
//! - layer norm with a bias, and the tanh GELU after a bias, each written as
//!   the f16 activations of the next product (`kernels/tower_rows.cuh`);
//! - the residual add with the product's bias, and the placing of the columns
//!   of a matrix computed in parts;
//! - the patches' bias and the learned position table resized to the grid of
//!   the image;
//! - the bias of query, key and value, the rotation of query and key by the
//!   patch's row and column, and their f16 rows per head;
//! - attention of every patch over every patch, no mask, on the tensor cores
//!   (`kernels/tower_attention.cuh`).
//!
//! Patches are in the merger's order (the 2 x 2 groups row by row, inside a
//! group (0,0), (0,1), (1,0), (1,1)).
//!
//! [`Kernels`]: crate::Kernels

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

/// The module `build.rs` compiled from `kernels/tower.cu`.
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tower.fatbin"));

/// Threads of the kernels that work on one row per block.
const ROW_THREADS: u32 = 256;
/// Threads per block of the kernels with one thread per value.
const VALUE_THREADS: u32 = 256;
/// The launch of the attention (`ARES_TOWER_ATT_*` in `kernels/tower_attention.cuh`):
/// 8 warps, 128 queries per block, two stages of 64 keys and values in shared
/// memory.
const ATTENTION_THREADS: u32 = 256;
const ATTENTION_QUERIES: u32 = 128;
const ATTENTION_KEYS: u32 = 64;
/// The most input rows one output row of [`TowerKernels::norm_f16`] holds.
const NORM_MAX_GROUP: u32 = 8;
/// The most pairs of a head [`TowerKernels::rotate_heads`] keeps angles for.
const ROTATE_MAX_PAIRS: u32 = 128;
/// Threads of [`TowerKernels::rotate_heads`]: at least one per pair.
const ROTATE_THREADS: u32 = 128;

/// The head sizes the attention has a kernel for, and its name.
const ATTENTION_HEADS: [(u32, &str); 1] = [(72, "ares_tower_attention_d72")];

/// The shape of the tower's attention over one image: `n_head` heads of
/// `head_dim` values for each of `n_patches` patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadShape {
    pub n_head: u32,
    pub head_dim: u32,
    pub n_patches: u32,
}

impl HeadShape {
    /// The f16 values of one head's row of query, key or value: the head's
    /// values, then zeros up to a multiple of 16 (the tensor cores' step).
    pub fn pitch(&self) -> u32 {
        self.head_dim.next_multiple_of(16)
    }

    /// The bytes of query, key and value of every head and patch in the form
    /// [`TowerKernels::rotate_heads`] writes and [`TowerKernels::attention`]
    /// reads: `[part][head][patch][pitch]` f16 values, the parts query, key and
    /// value.
    pub fn bytes(&self) -> u64 {
        3 * u64::from(self.n_head) * u64::from(self.n_patches) * u64::from(self.pitch()) * 2
    }

    /// The values of a row of the tower: all heads of a patch.
    pub fn width(&self) -> u32 {
        self.n_head * self.head_dim
    }
}

/// The grid of patches of one image and the learned position table: what
/// [`TowerKernels::positions`] resizes the table to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionGrid {
    /// Values of a patch row.
    pub n: u32,
    /// The table's side, in patches: `side * side` rows of `n` values.
    pub side: u32,
    /// Patches per line and lines of the image; both even.
    pub pw: u32,
    pub ph: u32,
    /// The source cells per output cell along a line and a column: the host's
    /// `(pw - 1) / (side - 1)` and `(ph - 1) / (side - 1)` in f32, as the
    /// reference forms them. Not used when the grid is the table's size.
    pub sfx: f32,
    pub sfy: f32,
}

/// The tower's kernels on one device.
#[derive(Clone, Debug)]
pub struct TowerKernels {
    device: Device,
    norm_f16: Function,
    gelu_f16: Function,
    residual: Function,
    place: Function,
    positions: Function,
    rotate_heads: Function,
    attention: Vec<(u32, Function)>,
}

fn invalid(what: &'static str, detail: String) -> Error {
    Error::Invalid { what, detail }
}

fn blocks_of(values: u64, threads: u32, what: &'static str) -> Result<u32> {
    u32::try_from(values.div_ceil(u64::from(threads)))
        .map_err(|_| invalid(what, format!("{values} values need more blocks than a grid has")))
}

impl TowerKernels {
    /// Loads the module on `device`.
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        Ok(Self {
            device: device.clone(),
            norm_f16: module.function("ares_tower_norm_f16")?,
            gelu_f16: module.function("ares_tower_gelu_f16")?,
            residual: module.function("ares_tower_residual")?,
            place: module.function("ares_tower_place")?,
            positions: module.function("ares_tower_positions")?,
            rotate_heads: module.function("ares_tower_rotate_heads")?,
            attention: ATTENTION_HEADS
                .iter()
                .map(|(d, name)| Ok((*d, module.function(name)?)))
                .collect::<Result<_>>()?,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether [`TowerKernels::attention`] has a kernel for heads of `head_dim`
    /// values.
    pub fn has_attention(head_dim: u32) -> bool {
        ATTENTION_HEADS.iter().any(|(d, _)| *d == head_dim)
    }

    fn row_shared(&self, what: &'static str, function: &Function, floats: u32) -> Result<u32> {
        let bytes = floats
            .checked_mul(4)
            .filter(|b| *b <= self.device.info().max_shared_per_block_optin)
            .ok_or_else(|| {
                invalid(what, format!("a row of {floats} values does not fit in shared memory"))
            })?;
        if bytes > 48 << 10 {
            function.allow_dynamic_shared(bytes)?;
        }
        Ok(bytes)
    }

    /// Layer norm of `n_rows * group` rows of `n` f32 values, written as
    /// `n_rows` rows of `n * group` f16 activations (the form of
    /// [`Kernels::prepare_prefill_activations`](crate::Kernels::prepare_prefill_activations)
    /// with the f16 path): every input row is normalised on its own,
    /// `(x - mean) / sqrt(variance + eps) * weight + bias`, its mean and
    /// variance in f32, and `group` consecutive ones are one output row (the
    /// merger reads four patches as one row). `n` must be a multiple of 4,
    /// `group` 1 to 8.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `x` must
    /// address `n_rows * group * n` f32 values, `weight` and `bias` `n` each,
    /// `out` the activation bytes of `n_rows` rows of `n * group` values, 16-byte
    /// aligned, not overlapping `x`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn norm_f16(
        &self,
        stream: &Stream,
        x: DevicePtr,
        weight: DevicePtr,
        bias: DevicePtr,
        out: DevicePtr,
        n: u32,
        group: u32,
        n_rows: u32,
        eps: f32,
    ) -> Result<()> {
        if n == 0 || !n.is_multiple_of(4) || group == 0 || group > NORM_MAX_GROUP {
            return Err(invalid(
                "tower norm",
                format!(
                    "rows of {n} values in groups of {group}: n a multiple of 4, groups 1 to 8"
                ),
            ));
        }
        if n_rows == 0 {
            return Ok(());
        }
        let shared = self.row_shared("tower norm", &self.norm_f16, n * group)?;
        let args = Args::new()
            .ptr(x)
            .ptr(weight)
            .ptr(bias)
            .ptr(out)
            .u32(n)
            .u32(group)
            .u32(n_rows)
            .f32(eps);
        // SAFETY: the argument list matches ares_tower_norm_f16; the ranges are the caller's
        // guarantee.
        unsafe { self.norm_f16.launch(stream, [n_rows, 1, 1], [ROW_THREADS, 1, 1], shared, args) }
    }

    /// The tanh GELU of `y + bias` for `n_rows` rows of `n` values, written as
    /// rows of `n_pad` f16 activations whose values from `n` on are zero (the
    /// input of a matrix padded with zero columns to `n_pad`).
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `y` must
    /// address `n_rows * n` f32 values, `bias` `n`, `out` the activation bytes of
    /// `n_rows` rows of `n_pad` values, 16-byte aligned, not overlapping `y`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gelu_f16(
        &self,
        stream: &Stream,
        y: DevicePtr,
        bias: DevicePtr,
        out: DevicePtr,
        n: u32,
        n_pad: u32,
        n_rows: u32,
    ) -> Result<()> {
        if n == 0 || n_pad < n {
            return Err(invalid("tower GELU", format!("rows of {n} values padded to {n_pad}")));
        }
        if n_rows == 0 {
            return Ok(());
        }
        let shared = self.row_shared("tower GELU", &self.gelu_f16, n_pad)?;
        let args = Args::new().ptr(y).ptr(bias).ptr(out).u32(n).u32(n_pad).u32(n_rows);
        // SAFETY: the argument list matches ares_tower_gelu_f16; the ranges are the caller's
        // guarantee.
        unsafe { self.gelu_f16.launch(stream, [n_rows, 1, 1], [ROW_THREADS, 1, 1], shared, args) }
    }

    /// The residual: `x += y + bias` for `n_rows` rows of `n` values (`n` a
    /// multiple of 4), the bias added to the product before the product is
    /// added to the residual.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `x` and `y`
    /// must address `n_rows * n` f32 values and `bias` `n`, all 16-byte aligned.
    pub unsafe fn residual(
        &self,
        stream: &Stream,
        x: DevicePtr,
        y: DevicePtr,
        bias: DevicePtr,
        n: u32,
        n_rows: u32,
    ) -> Result<()> {
        let values = u64::from(n) * u64::from(n_rows);
        if n == 0 || !n.is_multiple_of(4) || values > u64::from(u32::MAX) {
            return Err(invalid("tower residual", format!("{n_rows} rows of {n} values")));
        }
        if values == 0 {
            return Ok(());
        }
        let blocks = blocks_of(values / 4, VALUE_THREADS, "tower residual")?;
        let args = Args::new().ptr(x).ptr(y).ptr(bias).u32(n).u32(values as u32);
        // SAFETY: the argument list matches ares_tower_residual.
        unsafe { self.residual.launch(stream, [blocks, 1, 1], [VALUE_THREADS, 1, 1], 0, args) }
    }

    /// Columns of a product computed in parts, put in their place:
    /// `dst[r][at + i] = src[r][i] + bias[i]` for `n_rows` rows of `width` values
    /// in `src` and of `dst_width` in `dst`. `bias` is the part's own (`width`
    /// values) or [`DevicePtr::NULL`] for none.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `src` must
    /// address `n_rows * width` f32 values, `dst` `n_rows * dst_width`, `bias`
    /// `width` unless null; `src` and `dst` must not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn place(
        &self,
        stream: &Stream,
        src: DevicePtr,
        dst: DevicePtr,
        bias: DevicePtr,
        n_rows: u32,
        width: u32,
        dst_width: u32,
        at: u32,
    ) -> Result<()> {
        if at.checked_add(width).is_none_or(|end| end > dst_width) {
            return Err(invalid(
                "tower place",
                format!("{width} columns at {at} of rows of {dst_width}"),
            ));
        }
        let values = u64::from(n_rows) * u64::from(width);
        if values == 0 {
            return Ok(());
        }
        let blocks = blocks_of(values, VALUE_THREADS, "tower place")?;
        let args =
            Args::new().ptr(src).ptr(dst).ptr(bias).u32(n_rows).u32(width).u32(dst_width).u32(at);
        // SAFETY: the argument list matches ares_tower_place.
        unsafe { self.place.launch(stream, [blocks, 1, 1], [VALUE_THREADS, 1, 1], 0, args) }
    }

    /// The patches' bias and learned positions, in place: `x = (x + bias) +
    /// position` for the `grid.pw * grid.ph` patches of `x` (the patch product),
    /// the position of a patch being `table` resized bilinearly with aligned
    /// corners to the grid, or the table's own cell when the grid has its size.
    /// Every operation is the CPU reference's, in its order and in f32 without
    /// contraction.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `x` must
    /// address `pw * ph * n` f32 values, `bias` `n` and `table`
    /// `side * side * n`.
    pub unsafe fn positions(
        &self,
        stream: &Stream,
        x: DevicePtr,
        bias: DevicePtr,
        table: DevicePtr,
        grid: PositionGrid,
    ) -> Result<()> {
        let PositionGrid { n, side, pw, ph, sfx, sfy } = grid;
        let patches = u64::from(pw) * u64::from(ph);
        if n == 0 || side == 0 || pw % 2 != 0 || ph % 2 != 0 || patches > u64::from(u32::MAX) {
            return Err(invalid(
                "tower positions",
                format!("a grid of {pw} x {ph} patches of {n} values from a table of side {side}"),
            ));
        }
        if patches == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(x)
            .ptr(bias)
            .ptr(table)
            .u32(n)
            .u32(side)
            .u32(pw)
            .u32(ph)
            .f32(sfx)
            .f32(sfy);
        // SAFETY: the argument list matches ares_tower_positions.
        unsafe {
            self.positions.launch(stream, [patches as u32, 1, 1], [ROW_THREADS, 1, 1], 0, args)
        }
    }

    /// Query, key and value of patches `p0 .. p0 + n_rows` of an image of
    /// `shape.n_patches` patches (a grid of `nx` groups of 2 x 2 per line) as
    /// the attention reads them, from `qkv` (`n_rows` rows of query | key |
    /// value, each `shape.width()` values) and its `bias`: written into `heads`
    /// in the form of [`HeadShape::bytes`], query and key rotated. A head of
    /// `d` values has `d / 2` pairs `(k, k + d / 2)`; the first half of the
    /// pairs turns by the patch's row, the second by its column, pair `j` of a
    /// half by `position * step^j`, the angle built by multiplying on in f32 as
    /// the reference does.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `qkv` must
    /// address `n_rows * 3 * shape.width()` f32 values, `bias` `3 * shape.width()`
    /// and `heads` [`HeadShape::bytes`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rotate_heads(
        &self,
        stream: &Stream,
        qkv: DevicePtr,
        bias: DevicePtr,
        heads: DevicePtr,
        shape: HeadShape,
        p0: u32,
        n_rows: u32,
        nx: u32,
        step: f32,
    ) -> Result<()> {
        let pairs = shape.head_dim / 2;
        let fits = p0.checked_add(n_rows).is_some_and(|end| end <= shape.n_patches);
        if !shape.head_dim.is_multiple_of(4) || pairs > ROTATE_MAX_PAIRS || !fits {
            return Err(invalid(
                "tower rotation",
                format!(
                    "patches {p0} .. {p0} + {n_rows} of {} with heads of {} values (a multiple \
                     of 4, at most {})",
                    shape.n_patches,
                    shape.head_dim,
                    2 * ROTATE_MAX_PAIRS
                ),
            ));
        }
        if n_rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(qkv)
            .ptr(bias)
            .ptr(heads)
            .u32(shape.n_head)
            .u32(shape.head_dim)
            .u32(shape.pitch())
            .u32(shape.n_patches)
            .u32(p0)
            .u32(nx)
            .f32(step);
        // SAFETY: the argument list matches ares_tower_rotate_heads.
        unsafe { self.rotate_heads.launch(stream, [n_rows, 1, 1], [ROTATE_THREADS, 1, 1], 0, args) }
    }

    /// Attention of every patch over every patch, head by head:
    /// `out[p][h * d ..] = sum_j softmax_j(q_p . k_j * scale) v_j` from `heads`
    /// in the form of [`HeadShape::bytes`], written as rows of `shape.width()`
    /// f32 values. Products of f16 operands with f32 sums, the softmax in f32,
    /// the probabilities rounded to f16 for the second product.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `heads` must
    /// address [`HeadShape::bytes`] and `out` `n_patches * shape.width()` f32
    /// values; they must not overlap.
    pub unsafe fn attention(
        &self,
        stream: &Stream,
        heads: DevicePtr,
        out: DevicePtr,
        shape: HeadShape,
        scale: f32,
    ) -> Result<()> {
        let Some((_, function)) = self.attention.iter().find(|(d, _)| *d == shape.head_dim) else {
            return Err(invalid(
                "tower attention",
                format!(
                    "heads of {} values have no kernel (it has one for {:?})",
                    shape.head_dim,
                    ATTENTION_HEADS.map(|(d, _)| d)
                ),
            ));
        };
        if shape.n_patches == 0 || shape.n_head == 0 {
            return Ok(());
        }
        let shared = 2 * 2 * ATTENTION_KEYS * (shape.pitch() + 8) * 2;
        let scale_log2 = (f64::from(scale) * std::f64::consts::LOG2_E) as f32;
        let args =
            Args::new().ptr(heads).ptr(out).u32(shape.n_patches).u32(shape.n_head).f32(scale_log2);
        let grid = [shape.n_patches.div_ceil(ATTENTION_QUERIES), shape.n_head, 1];
        // SAFETY: the argument list matches ares_tower_attention_*.
        unsafe { function.launch(stream, grid, [ATTENTION_THREADS, 1, 1], shared, args) }
    }
}
