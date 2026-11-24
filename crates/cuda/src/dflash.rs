//! The kernels of the DFlash drafter (`kernels/dflash.cu`): its taps, the
//! dynamic convolution, the per-head norm and rotary turn, the ring of keys
//! and values per sequence, the attention of a block over its ring and itself,
//! the best candidates of each row, and the selector with the walk that makes
//! the drafts. The products are the weight kernels of [`Kernels`](crate::Kernels).

use crate::device::Device;
use crate::error::{Result, invalid};
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

/// The module `build.rs` compiled from `kernels/dflash.cu`.
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dflash.fatbin"));

/// Threads of the kernels with one thread per value.
const THREADS: u32 = 256;

/// The head size the attention takes.
pub const DFLASH_HEAD_DIM: u32 = 128;
/// The longest block the attention takes (a warp per row).
pub const DFLASH_MAX_BLOCK: u32 = 16;
/// The most candidates per position and the largest rank of the selector.
pub const DFLASH_MAX_TOP_K: u32 = 16;
pub const DFLASH_MAX_RANK: u32 = 256;

/// One sequence of a pass, as [`DflashKernels::attend`] reads it: its slot,
/// the position of its block's first row, and the first position its ring
/// holds a cell of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DflashSeq {
    pub slot: u32,
    pub position: u32,
    pub from: u32,
}

impl DflashSeq {
    pub const BYTES: usize = 16;

    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        for (i, v) in [self.slot, self.position, self.from, 0].into_iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// One sequence for [`DflashKernels::select`]: its newest token, the row of
/// the step record that takes its first draft (`u32::MAX` for none), how many
/// drafts its window takes, and where they go in the host's list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DflashSelect {
    pub newest: u32,
    pub record_row: u32,
    pub wanted: u32,
    pub place: u32,
}

impl DflashSelect {
    pub const BYTES: usize = 16;

    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        let words = [self.newest, self.record_row, self.wanted, self.place];
        for (i, v) in words.into_iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// The shape of the ring of keys and values: per slot and layer the keys of
/// `cells` cells, then their values, `kv` f16 values per cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    pub ptr: DevicePtr,
    pub layers: u32,
    pub cells: u32,
    pub kv: u32,
}

impl Ring {
    /// The bytes of one slot's ring.
    pub fn slot_bytes(layers: u32, cells: u32, kv: u32) -> u64 {
        u64::from(layers) * 2 * u64::from(cells) * u64::from(kv) * 2
    }
}

/// The selector's tables and sizes, for [`DflashKernels::select`].
#[derive(Clone, Copy, Debug)]
pub struct Selector {
    pub successor: DevicePtr,
    pub predecessor: DevicePtr,
    /// The ggml type id of the tables and the bytes of one row.
    pub ty: u32,
    pub row_bytes: u32,
    pub rank: u32,
    pub top_k: u32,
}

/// The drafter's kernels on one device.
#[derive(Clone, Debug)]
pub struct DflashKernels {
    tap: Function,
    conv: Function,
    head_norm_rope: Function,
    ring_write: Function,
    attend: Function,
    swiglu: Function,
    top: Function,
    select: Function,
}

fn blocks(values: u32) -> u32 {
    values.div_ceil(THREADS)
}

impl DflashKernels {
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        Ok(Self {
            tap: module.function("ares_dflash_tap")?,
            conv: module.function("ares_dflash_conv")?,
            head_norm_rope: module.function("ares_dflash_head_norm_rope")?,
            ring_write: module.function("ares_dflash_ring_write")?,
            attend: module.function("ares_dflash_attend")?,
            swiglu: module.function("ares_dflash_swiglu")?,
            top: module.function("ares_dflash_top")?,
            select: module.function("ares_dflash_select")?,
        })
    }

    /// `dst` row `r` (rows `dst_pitch` values apart) = the mean over `n`
    /// streams of `src` row `r`: stream `s` starts `stream_pitch` values after
    /// stream `s - 1`, its rows are `row_pitch` values apart. One stream is a
    /// copy.
    ///
    /// # Safety
    ///
    /// The ranges named must be f32 memory of the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tap(
        &self,
        stream: &Stream,
        src: DevicePtr,
        n: u32,
        stream_pitch: u64,
        row_pitch: u32,
        dst: DevicePtr,
        dst_pitch: u32,
        d: u32,
        rows: u32,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(src)
            .u32(n)
            .u64(stream_pitch)
            .u32(row_pitch)
            .ptr(dst)
            .u32(dst_pitch)
            .u32(d);
        // SAFETY: the arguments match the kernel; the ranges are the caller's.
        unsafe { self.tap.launch(stream, [blocks(d), rows, 1], [THREADS, 1, 1], 0, args) }
    }

    /// The dynamic convolution of `side` (0 before, 1 after) over `rows` rows
    /// of `d` values in blocks of `block`, into `y`: see `kernels/dflash.cu`.
    ///
    /// # Safety
    ///
    /// `x` and `y` hold `rows * d` f32, `dynamic` `rows` rows of
    /// `d / group * kernel * 2`, `base` `d * kernel * 2`; `y` overlaps nothing.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn conv(
        &self,
        stream: &Stream,
        x: DevicePtr,
        dynamic: DevicePtr,
        base: DevicePtr,
        y: DevicePtr,
        d: u32,
        group: u32,
        kernel: u32,
        side: u32,
        block: u32,
        rows: u32,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(x)
            .ptr(dynamic)
            .ptr(base)
            .ptr(y)
            .u32(d)
            .u32(group)
            .u32(kernel)
            .u32(side)
            .u32(block);
        // SAFETY: as documented.
        unsafe { self.conv.launch(stream, [blocks(d), rows, 1], [THREADS, 1, 1], 0, args) }
    }

    /// The norm of each head (when `norm` is some weights) and the rotary turn
    /// of its pairs by the row's position, in place.
    ///
    /// # Safety
    ///
    /// `x` holds `rows * heads * dim` f32, `positions` `rows` u32, `norm`
    /// `dim` f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn head_norm_rope(
        &self,
        stream: &Stream,
        x: DevicePtr,
        norm: Option<DevicePtr>,
        heads: u32,
        dim: u32,
        positions: DevicePtr,
        base: f32,
        eps: f32,
        rows: u32,
    ) -> Result<()> {
        if !dim.is_multiple_of(64) || dim > 512 {
            return Err(invalid("dflash heads", format!("heads of {dim} values")));
        }
        if rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(x)
            .ptr(norm.unwrap_or(DevicePtr::NULL))
            .u32(heads)
            .u32(dim)
            .ptr(positions)
            .f32(base)
            .f32(eps)
            .u32(u32::from(norm.is_some()));
        // SAFETY: as documented.
        unsafe { self.head_norm_rope.launch(stream, [heads, rows, 1], [dim / 2, 1, 1], 0, args) }
    }

    /// Writes rows of keys and values (`kv` f32 each) into `ring` at layer
    /// `layer`, the slot and position of each row read from `slots` and
    /// `positions` (a slot `u32::MAX` is skipped).
    ///
    /// # Safety
    ///
    /// The arrays hold `rows` entries; every slot named has its ring.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ring_write(
        &self,
        stream: &Stream,
        k: DevicePtr,
        v: DevicePtr,
        ring: &Ring,
        slots: DevicePtr,
        positions: DevicePtr,
        layer: u32,
        rows: u32,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(k)
            .ptr(v)
            .ptr(ring.ptr)
            .ptr(slots)
            .ptr(positions)
            .u32(layer)
            .u32(ring.layers)
            .u32(ring.cells)
            .u32(ring.kv);
        // SAFETY: as documented.
        unsafe { self.ring_write.launch(stream, [rows, 1, 1], [THREADS, 1, 1], 0, args) }
    }

    /// Attention of `sequences` blocks of `block` rows: queries `q` (rows of
    /// `heads * 128`), the block's keys and values `k`, `v` (rows of
    /// `kv_heads * 128`), the ring's cells of the last `window` positions, into
    /// `out` (rows of `heads * 128`). `seqs` holds a [`DflashSeq`] per block.
    ///
    /// # Safety
    ///
    /// Every buffer holds what is named above, on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attend(
        &self,
        stream: &Stream,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        ring: &Ring,
        out: DevicePtr,
        seqs: DevicePtr,
        layer: u32,
        window: u32,
        block: u32,
        heads: u32,
        kv_heads: u32,
        sequences: u32,
    ) -> Result<()> {
        if ring.kv != kv_heads * DFLASH_HEAD_DIM || block > DFLASH_MAX_BLOCK || block > 32 {
            return Err(invalid(
                "dflash attention",
                format!("{kv_heads} K/V heads in rows of {}, blocks of {block}", ring.kv),
            ));
        }
        if sequences == 0 {
            return Ok(());
        }
        let scale = 1.0 / (DFLASH_HEAD_DIM as f32).sqrt();
        let args = Args::new()
            .ptr(q)
            .ptr(k)
            .ptr(v)
            .ptr(ring.ptr)
            .ptr(out)
            .ptr(seqs)
            .u32(layer)
            .u32(ring.layers)
            .u32(ring.cells)
            .u32(window)
            .u32(block)
            .u32(heads)
            .u32(kv_heads)
            .f32(scale);
        // SAFETY: as documented.
        unsafe { self.attend.launch(stream, [sequences, heads, 1], [block * 32, 1, 1], 0, args) }
    }

    /// `out = silu(gate) * up` over `n` values.
    ///
    /// # Safety
    ///
    /// Each buffer holds `n` f32.
    pub unsafe fn swiglu(
        &self,
        stream: &Stream,
        gate: DevicePtr,
        up: DevicePtr,
        out: DevicePtr,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(gate).ptr(up).ptr(out).u32(n);
        // SAFETY: as documented.
        unsafe { self.swiglu.launch(stream, [blocks(n), 1, 1], [THREADS, 1, 1], 0, args) }
    }

    /// The best `k` of the first `n` columns of each of `rows` rows of
    /// logits (`stride` apart), the best first (equal values by column), into
    /// `cols` and `vals` (`k` per row).
    ///
    /// # Safety
    ///
    /// `logits` holds the rows, `cols` and `vals` `rows * k` entries.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn top(
        &self,
        stream: &Stream,
        logits: DevicePtr,
        n: u32,
        stride: u32,
        k: u32,
        cols: DevicePtr,
        vals: DevicePtr,
        rows: u32,
    ) -> Result<()> {
        if k == 0 || k > n {
            return Err(invalid("dflash candidates", format!("the best {k} of {n} columns")));
        }
        if rows == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(logits).u32(n).u32(stride).u32(k).ptr(cols).ptr(vals);
        // SAFETY: as documented.
        unsafe { self.top.launch(stream, [rows, 1, 1], [THREADS, 1, 1], 0, args) }
    }

    /// The selector and the walk for `sequences` passes of `positions` draft
    /// positions each: see `kernels/dflash.cu`. `seqs` holds a
    /// [`DflashSelect`] per sequence; `lattice` receives `k + k * k` f32 per
    /// sequence and position; the drafts go into `record_tokens` and `drafts`,
    /// their shares into `probs`.
    ///
    /// # Safety
    ///
    /// Every buffer holds what is named; the tables hold a row for every token
    /// the candidates and the newest tokens name.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn select(
        &self,
        stream: &Stream,
        cols: DevicePtr,
        vals: DevicePtr,
        gate: DevicePtr,
        head_tokens: DevicePtr,
        selector: &Selector,
        positions: u32,
        seqs: DevicePtr,
        lattice: DevicePtr,
        record_tokens: DevicePtr,
        drafts: DevicePtr,
        probs: DevicePtr,
        sequences: u32,
    ) -> Result<()> {
        if selector.top_k > DFLASH_MAX_TOP_K || selector.rank > DFLASH_MAX_RANK {
            return Err(invalid(
                "dflash selector",
                format!("{} candidates and a rank of {}", selector.top_k, selector.rank),
            ));
        }
        if sequences == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(cols)
            .ptr(vals)
            .ptr(gate)
            .ptr(head_tokens)
            .ptr(selector.successor)
            .ptr(selector.predecessor)
            .u32(selector.ty)
            .u32(selector.row_bytes)
            .u32(selector.rank)
            .u32(selector.top_k)
            .u32(positions)
            .ptr(seqs)
            .ptr(lattice)
            .ptr(record_tokens)
            .ptr(drafts)
            .ptr(probs);
        // SAFETY: as documented.
        unsafe { self.select.launch(stream, [sequences, 1, 1], [THREADS, 1, 1], 0, args) }
    }
}
