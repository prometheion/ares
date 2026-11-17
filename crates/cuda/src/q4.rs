//! The kernels of the `qwen4exp` architecture (Qwen3.8-Flash-Next, the preview
//! of Qwen4) beyond what [`LayerKernels`](crate::LayerKernels) and
//! [`AttentionKernels`](crate::AttentionKernels) have, from `kernels/q4.cu`:
//! the stream mixer, the device part of the hashed n-gram embedding, and the
//! indexer of the sparse attention with its block scores, the selection of
//! the cells a row attends to and the hand-over of that selection to the
//! attention kernels. `docs/reference/qwen4exp-architecture.md` (sections 3.3,
//! 3.4 and 3.6) defines what they compute.
//!
//! They follow the layer kernels: sizes are arguments, the rows of a step and
//! their sequences come from the step record ([`StepRecord`](crate::StepRecord)),
//! a window reads recurrent state and never writes it, and a commit writes
//! the accepted rows (`CommitSet::Deferred` at the end of the step for
//! segments with drafts, `CommitSet::Direct` right after the window for
//! segments whose rows are certain).
//!
//! # A read and a write of the stream mixer
//!
//! ```text
//! stream_norm        R -> xn                    (or stream_write_norm: the write of the block
//!                                                before and this norm in one pass)
//! product down       xn -> down                 Kernels::matvec_exact: f32 activations
//! squash             down -> lo
//! product up         lo -> gate
//! product inject     xn -> inj
//! mix_quant          xn, gate -> m              as quantized activations for the block's products
//! ... the block ...
//! stream_write       R += out * 2 sigmoid(inj / hc)
//! ```
//!
//! The three projections of a mixer multiply unquantized values, whatever type the
//! file stores them in (float matrices in most files, Q8_0 in some):
//! [`Kernels::matvec_exact`](crate::Kernels::matvec_exact) reads only the f32 section
//! of an activation buffer, so `stream_norm` and `squash` write plain f32: point them
//! at that section ([`quantized_activation_bytes`](crate::quantized_activation_bytes)
//! minus four bytes per value from the start) and no quantization runs in front of
//! those products. For more rows than the decode kernels take, the prefill
//! activations are prepared from the same f32 values. The fused read
//! ([`Q4Kernels::read_down`], [`Q4Kernels::read_up`]) takes every weight type too.
//!
//! # The hashed n-gram embedding: what the host supplies
//!
//! Per row of the step the stored bytes of the table rows of all heads, in
//! head order, in one buffer that goes to the device with the step
//! ([`NgramShape::fetched_row_bytes`]: 1440 bytes per row for Flash-Next's
//! IQ4_NL table; the table may be of any weight type). The host computes the
//! row ids from the row's token and the tokens of the two cells before it in
//! the same sequence (for a row inside a window those are the rows before it)
//! and reads the rows from the model file. Then:
//!
//! ```text
//! ngram_decode       fetched -> emb
//! products           emb -> key (ple_key), value (ple_value)
//! ngram_gate         key, value, R -> gated, nrm
//! ngram_window       R += gated + silu(conv(history, nrm))      the history is not written
//! ngram_commit       history <- nrm of the accepted rows
//! ```
//!
//! # An attention layer's indexer
//!
//! State: a **key pool** per attention layer (a slot holds the pooled keys of
//! the `128 / B` blocks of one table entry, in [`KeyFormat`]), addressed
//! through a `u32` array parallel to the attention tables (same index, the
//! value is the slot in the key pool); and per sequence slot the **carried
//! state** ([`IndexerShape::carry_len`] f32: the raw keys of the cells of the
//! unfinished block and the rotation of its first cell).
//!
//! ```text
//! products           m -> iq (indexer.q_proj), raw (indexer.k_proj)
//! indexer_window     iq -> u (normed, rotated); pooled keys of the blocks the rows complete
//! indexer_commit(Direct)
//! block_scores       u, key pool -> scores      rows past the dense limit
//! select             scores -> lists of cell ranges per row, written into the attention plan
//! hand_over          lists -> the reserved items of the plan
//! AttentionKernels::attend
//! ... end of the step:
//! indexer_commit(Deferred)
//! ```
//!
//! The cell and the table of a row are read from the attention plan's row
//! array ([`AttnRow`]). All launches are `unsafe`: the device trusts the
//! addresses it is given.

use ares_core::GgmlType;
use ares_core::ggml_type::ALL_TYPES;

use crate::attention::{AttnItem, AttnRow, ENTRY_CELLS};
use crate::device::Device;
use crate::error::{Error, Result};
use crate::layers::{CommitSet, StepDims};
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4.fatbin"));

/// The fused read of the stream mixer for the quantized types (`kernels/q4_reads.cu`).
static READS_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_reads.fatbin"));

// The kernels read these two types of the attention module by offset.
const _: () = {
    assert!(std::mem::size_of::<AttnRow>() == 16);
    assert!(std::mem::offset_of!(AttnRow, table) == 0 && std::mem::offset_of!(AttnRow, pos) == 4);
    assert!(std::mem::size_of::<AttnItem>() == 48);
    assert!(std::mem::offset_of!(AttnItem, ranges_off) == 16);
    assert!(std::mem::offset_of!(AttnItem, n_ranges) == 20);
    assert!(std::mem::offset_of!(AttnItem, total) == 24);
};

/// The most blocks of a sequence the selection handles: 262 144 cells of 4.
pub const SELECT_MAX_BLOCKS: u32 = 65_536;
/// The most pieces a list is cut into by [`Q4Kernels::hand_over`].
pub const HAND_OVER_MAX_PIECES: u32 = 64;

fn invalid(what: &'static str, detail: String) -> Error {
    Error::Invalid { what, detail }
}

fn check_align(what: &'static str, ptrs: &[DevicePtr], align: u64) -> Result<()> {
    match ptrs.iter().find(|p| !p.as_raw().is_multiple_of(align)) {
        Some(p) => {
            Err(invalid(what, format!("address {:#x} is not aligned to {align} bytes", p.as_raw())))
        }
        None => Ok(()),
    }
}

/// How the pooled keys of the indexer are stored in the key pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyFormat {
    /// The definition: 4 bytes per value, 1536 bytes per cell over the 12
    /// attention layers of Flash-Next.
    F32,
    /// Half of that. A key is rounded once when its block completes (by up
    /// to 4e-4 of its largest value), which can swap blocks whose scores are
    /// that close: on random keys 5 of 28 714 selected cells differed from
    /// the selection with f32 keys.
    F16,
}

impl KeyFormat {
    /// Bytes per value.
    pub const fn bytes(self) -> u64 {
        match self {
            KeyFormat::F32 => 4,
            KeyFormat::F16 => 2,
        }
    }
}

/// How a row past the dense limit picks its cells (difference 1 of the
/// architecture document).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SelectionRule {
    /// The tail, then cells of the best blocks until `top_k + block - 1`
    /// cells are selected; the last block taken may give only its first cells.
    CellBudget = 0,
    /// The best `top_k / block` whole blocks and the tail.
    WholeBlocks = 1,
}

/// The shapes of the indexer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexerShape {
    /// Indexer query heads (4).
    pub n_head: u32,
    /// The width of a query head and of a key (128).
    pub key_dim: u32,
    /// Cells per block (4).
    pub block: u32,
    /// The selection budget in cells (2048).
    pub top_k: u32,
    /// The rotated values of a head (64).
    pub rope_dim: u32,
}

impl IndexerShape {
    /// The largest number of cells a row attends to without selection.
    pub const fn dense_limit(&self) -> u32 {
        self.top_k + self.block - 1
    }

    /// Blocks per table entry of 128 cells.
    pub const fn blocks_per_entry(&self) -> u32 {
        ENTRY_CELLS / self.block
    }

    /// Bytes of a slot of the key pool: the pooled keys of one table entry.
    pub const fn key_slot_bytes(&self, format: KeyFormat) -> u64 {
        self.blocks_per_entry() as u64 * self.key_dim as u64 * format.bytes()
    }

    /// The f32 values of the carried state of one sequence slot and layer:
    /// the raw keys of the cells of the unfinished block (`block - 1` of
    /// them at most), then the rotation of its first cell.
    pub const fn carry_len(&self) -> u64 {
        (self.block as u64 - 1) * self.key_dim as u64 + self.rope_dim as u64
    }

    /// The most ranges the two lists of a row have together when the row's
    /// table changes between converted and exact entries `kind_changes`
    /// times: the capacity [`Q4Kernels::select`] needs per row.
    pub const fn max_ranges(&self, kind_changes: u32) -> u32 {
        self.top_k / self.block + 2 + kind_changes
    }

    fn config(&self, format: KeyFormat) -> u64 {
        u64::from(self.key_dim)
            | u64::from(self.n_head) << 16
            | u64::from(self.block) << 24
            | u64::from(self.rope_dim) << 32
            | (format as u64) << 48
    }

    fn check(&self) -> Result<()> {
        let bad = |detail: String| Err(invalid("indexer shape", detail));
        if self.key_dim == 0 || !self.key_dim.is_multiple_of(32) || self.key_dim > 128 {
            return bad(format!("key width {} (a multiple of 32, at most 128)", self.key_dim));
        }
        if self.n_head == 0 || self.n_head > 4 {
            return bad(format!("{} query heads (1 to 4)", self.n_head));
        }
        if !matches!(self.block, 1 | 2 | 4) {
            return bad(format!("blocks of {} cells (1, 2 or 4)", self.block));
        }
        if self.top_k == 0 || !self.top_k.is_multiple_of(self.block) || self.top_k > 32_768 {
            return bad(format!("a budget of {} cells", self.top_k));
        }
        if !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.key_dim {
            return bad(format!("{} rotated values of {}", self.rope_dim, self.key_dim));
        }
        Ok(())
    }
}

/// What the attention plan reserved for a row whose lists are made on the
/// device, as [`Q4Kernels::hand_over`] reads it (32 bytes).
///
/// The row's lists occupy `3 * cap` words of the plan's `ranges` from
/// `ranges_off` on. The items of a list are contiguous in the plan's item
/// array, `kv_heads` items per piece, piece-major; the item of piece `i` and
/// KV head `h` is `item + i * kv_heads + h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ListHandOver {
    /// Index in the plan's `ranges` (in `u32`) of the row's lists.
    pub ranges_off: u32,
    /// The first item of the converted list.
    pub item_conv: u32,
    /// Pieces reserved for the converted list (0 when the sequence has no
    /// converted entry).
    pub pieces_conv: u32,
    /// The first item of the exact list.
    pub item_exact: u32,
    /// Pieces reserved for the exact list.
    pub pieces_exact: u32,
    /// Items per piece: the KV heads in the pools.
    pub kv_heads: u32,
}

impl ListHandOver {
    /// Bytes of one record on the device.
    pub const BYTES: usize = 32;

    /// The records as the bytes to upload.
    pub fn to_bytes(hands: &[ListHandOver]) -> Vec<u8> {
        hands
            .iter()
            .flat_map(|h| {
                [
                    h.ranges_off,
                    h.item_conv,
                    h.pieces_conv,
                    h.item_exact,
                    h.pieces_exact,
                    h.kv_heads,
                    0,
                    0,
                ]
            })
            .flat_map(u32::to_le_bytes)
            .collect()
    }
}

/// The shapes of the hashed n-gram block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NgramShape {
    /// Residual streams.
    pub n_stream: u32,
    /// The width of a stream.
    pub n_embd: u32,
    /// Table rows per position: the hash heads (16).
    pub heads: u32,
    /// Values of a table row (160).
    pub row_dim: u32,
    /// The type of the table's rows as the file stores them (IQ4_NL in the real file): any
    /// weight type whose blocks divide `row_dim`.
    pub ty: GgmlType,
    /// Taps of the convolution (4).
    pub conv_kernel: u32,
    /// Its dilation: the n-gram size (3).
    pub dilation: u32,
}

impl NgramShape {
    /// The width of the residual.
    pub const fn wide(&self) -> u32 {
        self.n_stream * self.n_embd
    }

    /// The cells of history the convolution reads.
    pub const fn back(&self) -> u32 {
        (self.conv_kernel - 1) * self.dilation
    }

    /// The f32 values of the history of one sequence slot.
    pub const fn history_len(&self) -> u64 {
        self.back() as u64 * self.wide() as u64
    }

    /// Bytes of one table row as the file stores it.
    pub const fn table_row_bytes(&self) -> u64 {
        let (values, bytes) = self.ty.block();
        (self.row_dim / values) as u64 * bytes as u64
    }

    /// Bytes the host supplies per row of a step: the table rows of all heads.
    pub const fn fetched_row_bytes(&self) -> u64 {
        self.heads as u64 * self.table_row_bytes()
    }

    fn check(&self) -> Result<()> {
        let bad = |detail: String| Err(invalid("n-gram shape", detail));
        if self.n_stream == 0 || self.n_embd == 0 {
            return bad(format!("{} streams of {}", self.n_stream, self.n_embd));
        }
        if !self.ty.is_weight()
            || self.row_dim == 0
            || !self.row_dim.is_multiple_of(self.ty.block_values())
            || self.heads * self.row_dim != self.n_embd
        {
            return bad(format!(
                "{} rows of {} values of {} for a width of {}",
                self.heads, self.row_dim, self.ty, self.n_embd
            ));
        }
        if self.conv_kernel < 2 || self.conv_kernel > 8 || self.dilation == 0 || self.back() > 32 {
            return bad(format!(
                "{} taps with dilation {} (2 to 8 taps, a history of at most 32 cells)",
                self.conv_kernel, self.dilation
            ));
        }
        Ok(())
    }
}

/// The `qwen4exp` kernels on one device.
#[derive(Clone, Debug)]
pub struct Q4Kernels {
    device: Device,
    stream: Function,
    squash: Function,
    gate_scale: Function,
    mix_quant: Function,
    indexer_window: Function,
    indexer_commit: Function,
    block_scores: Function,
    select: Function,
    hand_over: Function,
    ngram_decode: Function,
    ngram_gate: Function,
    ngram_window: Function,
    ngram_commit: Function,
    /// The two launches of the fused read per type of the matrix they read (the float
    /// types share one pair, which takes the type as an argument), with the most threads
    /// a block of each can have on this device.
    reads: Vec<Read>,
    /// The pair that decodes every type by its id at run time, and `inject` of any type,
    /// with the same.
    read_any: Read,
    swap: Function,
    route_split: Function,
    route_recv: Function,
}

fn check_streams(what: &'static str, hc: u32, n: u32) -> Result<()> {
    if hc == 0 || n == 0 || !n.is_multiple_of(4) {
        return Err(invalid(what, format!("{hc} streams of {n} values (a multiple of 4)")));
    }
    Ok(())
}

impl Q4Kernels {
    /// Loads the kernels on `device`.
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        Ok(Self {
            device: device.clone(),
            stream: module.function("ares_q4_stream")?,
            squash: module.function("ares_q4_squash")?,
            gate_scale: module.function("ares_q4_gate_scale")?,
            mix_quant: module.function("ares_q4_mix_quant")?,
            indexer_window: module.function("ares_q4_indexer_window")?,
            indexer_commit: module.function("ares_q4_indexer_commit")?,
            block_scores: module.function("ares_q4_block_scores")?,
            select: module.function("ares_q4_select")?,
            hand_over: module.function("ares_q4_hand_over")?,
            ngram_decode: module.function("ares_q4_ngram_decode")?,
            ngram_gate: module.function("ares_q4_ngram_gate")?,
            ngram_window: module.function("ares_q4_ngram_window")?,
            ngram_commit: module.function("ares_q4_ngram_commit")?,
            reads: Self::load_reads(&module, &Module::load(device, READS_IMAGE)?)?,
            read_any: Read::load(&module, "_any")?,
            swap: module.function("ares_q4_swap")?,
            route_split: module.function("ares_q4_route_split")?,
            route_recv: module.function("ares_q4_route_recv")?,
        })
    }

    /// The device the kernels are loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The pairs of the fused read for every weight type: those of the float types and the
    /// one that decodes by the type at run time from `module`, those of [`INLINED_READS`]
    /// from `typed`, by the names `kernels/q4_reads.cu` gives them.
    fn load_reads(module: &Module, typed: &Module) -> Result<Vec<Read>> {
        let float = Read::load(module, "")?;
        let any = Read::load(module, "_any")?;
        ALL_TYPES
            .into_iter()
            .filter(|ty| ty.is_weight())
            .map(|ty| {
                let read = if is_float(ty) {
                    float.clone()
                } else if INLINED_READS.contains(&ty) {
                    Read::load(typed, &format!("_{}", ty.name()))?
                } else {
                    any.clone()
                };
                Ok(Read { ty, ..read })
            })
            .collect()
    }

    /// The pair of the fused read for a matrix of `ty`.
    fn reads_of(&self, ty: GgmlType) -> Result<&Read> {
        self.reads.iter().find(|r| r.ty == ty).ok_or_else(|| {
            invalid("read", format!("a mixer matrix of {ty}, which holds no weights"))
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_stream(
        &self,
        stream: &Stream,
        r: DevicePtr,
        write: Option<(DevicePtr, DevicePtr)>,
        norm: Option<(DevicePtr, DevicePtr)>,
        rows: u32,
        hc: u32,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        check_streams("stream mixer", hc, n)?;
        let (block_out, inj) = write.unwrap_or((DevicePtr::NULL, DevicePtr::NULL));
        let (weight, xn) = norm.unwrap_or((DevicePtr::NULL, DevicePtr::NULL));
        check_align("stream mixer", &[r, block_out, weight, xn], 16)?;
        let blocks = rows
            .checked_mul(hc)
            .ok_or_else(|| invalid("stream mixer", format!("{rows} rows of {hc} streams")))?;
        if blocks == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(r)
            .ptr(block_out)
            .ptr(inj)
            .ptr(weight)
            .ptr(xn)
            .u32(rows)
            .u32(hc)
            .u32(n)
            .f32(eps);
        // SAFETY: the argument list matches ares_q4_stream; the ranges are the caller's guarantee.
        unsafe { self.stream.launch(stream, [blocks, 1, 1], [256, 1, 1], 0, args) }
    }

    /// The norm of a read: `xn[row][c] = rms(r[row][c]) * norm[c]` for the
    /// `hc` streams of `n` values of every row, the RMS over the stream's own
    /// values. `norm` has a weight per value and stream (`hc * n`).
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `r` and
    /// `xn` must address `rows * hc * n` f32 values, `norm` `hc * n`, all
    /// 16-byte aligned; `xn` must not overlap the others.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn stream_norm(
        &self,
        stream: &Stream,
        r: DevicePtr,
        norm: DevicePtr,
        xn: DevicePtr,
        rows: u32,
        hc: u32,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe { self.launch_stream(stream, r, None, Some((norm, xn)), rows, hc, n, eps) }
    }

    /// The write of a block's output back to the streams:
    /// `r[row][c] += block_out[row] * 2 * sigmoid(inj[row][c] / hc)`.
    ///
    /// # Safety
    ///
    /// As [`Q4Kernels::stream_norm`] for `r`; `block_out` must address
    /// `rows * n` f32 values (16-byte aligned) and `inj` `rows * hc`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn stream_write(
        &self,
        stream: &Stream,
        r: DevicePtr,
        block_out: DevicePtr,
        inj: DevicePtr,
        rows: u32,
        hc: u32,
        n: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe { self.launch_stream(stream, r, Some((block_out, inj)), None, rows, hc, n, 0.0) }
    }

    /// [`Q4Kernels::stream_write`] and the [`Q4Kernels::stream_norm`] of the
    /// read that follows it, in one pass over the residual.
    ///
    /// # Safety
    ///
    /// As the two.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn stream_write_norm(
        &self,
        stream: &Stream,
        r: DevicePtr,
        block_out: DevicePtr,
        inj: DevicePtr,
        norm: DevicePtr,
        xn: DevicePtr,
        rows: u32,
        hc: u32,
        n: u32,
        eps: f32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe {
            self.launch_stream(
                stream,
                r,
                Some((block_out, inj)),
                Some((norm, xn)),
                rows,
                hc,
                n,
                eps,
            )
        }
    }

    /// The low-rank activation of a read: `lo = silu(down / hc)` over
    /// `n_values` values (rows times the mixer's rank).
    ///
    /// # Safety
    ///
    /// `down` and `lo` must address `n_values` f32 values on the stream's
    /// device, alive until the launch has run. In place is allowed.
    pub unsafe fn squash(
        &self,
        stream: &Stream,
        down: DevicePtr,
        lo: DevicePtr,
        n_values: u32,
        hc: u32,
    ) -> Result<()> {
        if hc == 0 {
            return Err(invalid("squash", "no streams".into()));
        }
        if n_values == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(down).ptr(lo).u32(n_values).f32(hc as f32);
        // SAFETY: the argument list matches ares_q4_squash.
        unsafe { self.squash.launch(stream, [n_values.div_ceil(256), 1, 1], [256, 1, 1], 0, args) }
    }

    /// The scalar gate of the shared expert of the mixture of experts:
    /// `out[row] = y[row] * sigmoid(s[row])` for `rows` rows of `n` values,
    /// with one gate value per row.
    ///
    /// # Safety
    ///
    /// `y` and `out` must address `rows * n` f32 values and `s` `rows`, on
    /// the stream's device, alive until the launch has run. In place
    /// (`out == y`) is allowed.
    pub unsafe fn gate_scale(
        &self,
        stream: &Stream,
        y: DevicePtr,
        s: DevicePtr,
        out: DevicePtr,
        rows: u32,
        n: u32,
    ) -> Result<()> {
        let n_values = rows
            .checked_mul(n)
            .ok_or_else(|| invalid("gate", format!("{rows} rows of {n} values")))?;
        if n_values == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(y).ptr(s).ptr(out).u32(rows).u32(n);
        // SAFETY: the argument list matches ares_q4_gate_scale.
        unsafe {
            self.gate_scale.launch(stream, [n_values.div_ceil(256), 1, 1], [256, 1, 1], 0, args)
        }
    }

    /// The gated mean of a read, `m[row] = (1 / hc) * sum_c xn[row][c] *
    /// sigmoid(gate[row][c])`, written as quantized activations over
    /// `rows * n` values (the layout of `Kernels::quantize_activations`,
    /// [`quantized_activation_bytes`](crate::quantized_activation_bytes) bytes;
    /// the f32 values are its last section). `n` is a multiple of 32.
    ///
    /// # Safety
    ///
    /// `xn` and `gate` must address `rows * hc * n` f32 values and `act` the
    /// bytes above, 16-byte aligned, on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mix_quant(
        &self,
        stream: &Stream,
        xn: DevicePtr,
        gate: DevicePtr,
        act: DevicePtr,
        rows: u32,
        hc: u32,
        n: u32,
    ) -> Result<()> {
        if hc == 0 || n == 0 || !n.is_multiple_of(32) {
            return Err(invalid("mix", format!("{hc} streams of {n} values (a multiple of 32)")));
        }
        check_align("mix", &[xn, gate, act], 16)?;
        let chunks = u64::from(rows) * u64::from(n / 4);
        let blocks = u32::try_from(chunks.div_ceil(256))
            .map_err(|_| invalid("mix", format!("{rows} rows of {n} values")))?;
        if blocks == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(xn).ptr(gate).ptr(act).u32(rows).u32(hc).u32(n);
        // SAFETY: the argument list matches ares_q4_mix_quant.
        unsafe { self.mix_quant.launch(stream, [blocks, 1, 1], [256, 1, 1], 0, args) }
    }

    /// The indexer of one attention layer for the rows of a step: the queries
    /// `u[row][e] = rope(rms(iq[row][e]) * q_norm)` and the pooled keys of the
    /// blocks the segments complete (the mean of a block's raw keys through
    /// the RMS norm with `k_norm` and the rotation of the block's first cell),
    /// written into the key pool, except those of a replay segment (flagged
    /// in the record: its blocks are cached). The carried state is read, not
    /// written.
    ///
    /// - `attn_rows`: the attention plan's rows ([`AttnRow`] per row of the step).
    /// - `iq` `[row][n_head][key_dim]` and `raw` `[row][key_dim]`: the two
    ///   projections. `u_out` like `iq`; in place is allowed.
    /// - `rope`: the table of `LayerKernels::rope_table` for `shape.rope_dim`.
    /// - `carry`: the carried state, `carry_stride` f32 per sequence slot
    ///   (at least [`IndexerShape::carry_len`]).
    /// - `key_table`: per table entry the slot in `key_pool`.
    ///
    /// # Safety
    ///
    /// Every pointer must address live memory of these sizes on the stream's
    /// device (f32 arrays 4-byte aligned, `carry` 8-byte aligned), and the
    /// key table must name a slot of `key_pool` for every entry the rows'
    /// blocks lie in.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn indexer_window(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: IndexerShape,
        format: KeyFormat,
        eps: f32,
        attn_rows: DevicePtr,
        iq: DevicePtr,
        raw: DevicePtr,
        q_norm: DevicePtr,
        k_norm: DevicePtr,
        rope: DevicePtr,
        u_out: DevicePtr,
        carry: DevicePtr,
        carry_stride: u64,
        key_table: DevicePtr,
        key_pool: DevicePtr,
    ) -> Result<()> {
        shape.check()?;
        check_align("indexer", &[carry], 8)?;
        if carry_stride < shape.carry_len() || !carry_stride.is_multiple_of(2) {
            return Err(invalid(
                "indexer",
                format!(
                    "a carried state of {carry_stride} values, {} are needed",
                    shape.carry_len()
                ),
            ));
        }
        if dims.n_rows == 0 {
            return Ok(());
        }
        let max_new_blocks = dims.max_seg_rows.div_ceil(shape.block);
        let items = u64::from(dims.n_rows) * u64::from(shape.n_head)
            + u64::from(dims.n_segs) * u64::from(max_new_blocks);
        let blocks = u32::try_from(items.div_ceil(8))
            .map_err(|_| invalid("indexer", "too many rows".into()))?;
        let args = Args::new()
            .ptr(step)
            .ptr(attn_rows)
            .ptr(iq)
            .ptr(raw)
            .ptr(q_norm)
            .ptr(k_norm)
            .ptr(rope)
            .ptr(u_out)
            .ptr(carry)
            .u64(carry_stride)
            .ptr(key_table)
            .ptr(key_pool)
            .u64(shape.config(format))
            .f32(eps)
            .u32(max_new_blocks.max(1));
        // SAFETY: the argument list matches ares_q4_indexer_window.
        unsafe { self.indexer_window.launch(stream, [blocks, 1, 1], [256, 1, 1], 0, args) }
    }

    /// The commit of the carried indexer state of one attention layer: the
    /// raw keys of the cells of the block that is unfinished after the
    /// accepted rows, and the rotation of its first cell. `which` selects the
    /// segments as in `LayerKernels::conv_commit`.
    ///
    /// # Safety
    ///
    /// `raw` and `rope` must still hold what the window of the step read;
    /// the rest as [`Q4Kernels::indexer_window`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn indexer_commit(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: IndexerShape,
        which: CommitSet,
        attn_rows: DevicePtr,
        raw: DevicePtr,
        rope: DevicePtr,
        carry: DevicePtr,
        carry_stride: u64,
    ) -> Result<()> {
        shape.check()?;
        check_align("indexer", &[carry], 8)?;
        if dims.n_segs == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(step)
            .ptr(attn_rows)
            .ptr(raw)
            .ptr(rope)
            .ptr(carry)
            .u64(carry_stride)
            .u64(shape.config(KeyFormat::F32))
            .u32(which as u32);
        // SAFETY: the argument list matches ares_q4_indexer_commit.
        unsafe { self.indexer_commit.launch(stream, [dims.n_segs, 1, 1], [128, 1, 1], 0, args) }
    }

    /// The scores of the blocks the rows `row0 .. row0 + n_rows` of the step
    /// see: `scores[i][j] = sum_e max(0, u[row0 + i][e] . P[j])` for the
    /// complete blocks `j` of the row's sequence, with the pooled keys read
    /// from the key pool in its format. A row within the dense limit scores
    /// nothing and its part of `scores` is not written. `max_blocks` is the
    /// distance between the rows in `scores` and at least the blocks of the
    /// longest row.
    ///
    /// A long step is scored in pieces of rows to bound the scratch:
    /// `n_rows * max_blocks * 4` bytes per launch.
    ///
    /// # Safety
    ///
    /// `attn_rows` and `u` (`[row][n_head][key_dim]` f32) must cover the rows
    /// of the launch, `scores` `n_rows * max_blocks` f32, and the key table
    /// and pool every block the rows see, on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn block_scores(
        &self,
        stream: &Stream,
        shape: IndexerShape,
        format: KeyFormat,
        attn_rows: DevicePtr,
        u: DevicePtr,
        key_table: DevicePtr,
        key_pool: DevicePtr,
        scores: DevicePtr,
        row0: u32,
        n_rows: u32,
        max_blocks: u32,
    ) -> Result<()> {
        shape.check()?;
        if n_rows == 0 || max_blocks == 0 {
            return Ok(());
        }
        // Blocks per thread block: twice the thread blocks the device holds at once (six of 256
        // threads are resident per multiprocessor) when one row scores a long sequence, so
        // that no multiprocessor idles while the last ones finish; fewer per row when many
        // rows share the device.
        let wanted = 12 * self.device.info().multiprocessors;
        let span = max_blocks.div_ceil((wanted / n_rows).max(1)).max(64).next_multiple_of(8);
        let args = Args::new()
            .ptr(attn_rows)
            .ptr(u)
            .ptr(key_table)
            .ptr(key_pool)
            .ptr(scores)
            .u64(shape.config(format))
            .u32(row0)
            .u32(max_blocks)
            .u32(span)
            .u32(shape.dense_limit());
        // SAFETY: the argument list matches ares_q4_block_scores.
        unsafe {
            self.block_scores.launch(
                stream,
                [max_blocks.div_ceil(span), n_rows, 1],
                [256, 1, 1],
                0,
                args,
            )
        }
    }

    /// The cells the rows `row0 .. row0 + n_rows` attend to, as two ascending
    /// lists of ranges per row: the cells in converted entries of the
    /// attention tables, then those in exact entries. A list is `(start,
    /// end, cells of the list before the range)` triples of `u32`, the form
    /// of an attention item's ranges; no range crosses between entries of
    /// different kinds.
    ///
    /// - `scores`: from [`Q4Kernels::block_scores`] for the same rows, `score_stride` apart.
    /// - `tables`: the attention tables, or `None` when every entry is of one
    ///   kind: then all cells are in the first list.
    /// - `lists`: `cap` triples per row of the launch, the converted list
    ///   first and the exact list right behind it.
    /// - `counts`: 4 `u32` per row of the launch: ranges and cells of the
    ///   converted list, ranges and cells of the exact list. A row that
    ///   needs more than `cap` triples gets the counts and the first `cap`
    ///   triples ([`IndexerShape::max_ranges`] is enough).
    ///
    /// A row within the dense limit gets all its cells; its scores are not read.
    ///
    /// # Safety
    ///
    /// The buffers must be live on the stream's device and of these sizes;
    /// no row of the launch may see more than [`SELECT_MAX_BLOCKS`] blocks.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn select(
        &self,
        stream: &Stream,
        shape: IndexerShape,
        rule: SelectionRule,
        attn_rows: DevicePtr,
        scores: DevicePtr,
        tables: Option<DevicePtr>,
        lists: DevicePtr,
        counts: DevicePtr,
        row0: u32,
        n_rows: u32,
        score_stride: u32,
        cap: u32,
    ) -> Result<()> {
        shape.check()?;
        if n_rows == 0 {
            return Ok(());
        }
        if cap == 0 {
            return Err(invalid("selection", "no room for ranges".into()));
        }
        let args = Args::new()
            .ptr(attn_rows)
            .ptr(scores)
            .ptr(tables.unwrap_or(DevicePtr::NULL))
            .ptr(lists)
            .ptr(counts)
            .u64(shape.config(KeyFormat::F32))
            .u32(row0)
            .u32(score_stride)
            .u32(shape.top_k)
            .u32(rule as u32)
            .u32(cap);
        // SAFETY: the argument list matches ares_q4_select.
        unsafe { self.select.launch(stream, [n_rows, 1, 1], [256, 1, 1], 0, args) }
    }

    /// Hands the lists of [`Q4Kernels::select`] to the attention kernels. The
    /// select launch must have written into the plan's `ranges` (its `lists`
    /// at `ranges + 4 * hands[0].ranges_off`, the rows `3 * cap` words apart
    /// as the records say). Each list is cut into its reserved pieces of about
    /// equal numbers of cells at range boundaries, and the items of a piece
    /// get their `ranges_off`, `n_ranges` and `total`; a piece without a range
    /// gets `total = 0`.
    ///
    /// - `hands`: one [`ListHandOver`] per row of the select launch, in its order.
    /// - `counts`: the counts the select launch wrote.
    /// - `ranges`, `items`: the plan's arrays on the device ([`AttnItem`]).
    ///
    /// # Safety
    ///
    /// The records must name items and ranges inside the plan's arrays, on
    /// the stream's device, and no other launch may read the plan in between.
    pub unsafe fn hand_over(
        &self,
        stream: &Stream,
        hands: DevicePtr,
        counts: DevicePtr,
        ranges: DevicePtr,
        items: DevicePtr,
        n_rows: u32,
    ) -> Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(hands).ptr(counts).ptr(ranges).ptr(items);
        // SAFETY: the argument list matches ares_q4_hand_over.
        unsafe { self.hand_over.launch(stream, [n_rows, 1, 1], [256, 1, 1], 0, args) }
    }

    /// Decodes the fetched table rows of the n-gram embedding: `fetched` holds
    /// [`NgramShape::fetched_row_bytes`] bytes per row of the step (the table
    /// rows of the heads in order, in the table's type), `emb` gets
    /// `n_rows * n_embd` f32.
    ///
    /// # Safety
    ///
    /// `fetched` (4-byte aligned) and `emb` must address these sizes on the
    /// stream's device.
    pub unsafe fn ngram_decode(
        &self,
        stream: &Stream,
        shape: NgramShape,
        fetched: DevicePtr,
        emb: DevicePtr,
        n_rows: u32,
    ) -> Result<()> {
        shape.check()?;
        // An F32 table is read in words, the f16 fields of the others in halves: every row
        // of every type keeps that alignment (the blocks of MXFP4, 17 bytes, are read in
        // bytes).
        check_align("n-gram rows", &[fetched], 4)?;
        let row_bytes = shape.table_row_bytes();
        let n_values = n_rows
            .checked_mul(shape.n_embd)
            .ok_or_else(|| invalid("n-gram rows", format!("{n_rows} rows")))?;
        if n_values == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(fetched)
            .ptr(emb)
            .u32(shape.ty.id())
            .u32(shape.row_dim)
            .u32(row_bytes as u32)
            .u32(n_values);
        // SAFETY: the argument list matches ares_q4_ngram_decode.
        unsafe {
            self.ngram_decode.launch(stream, [n_values.div_ceil(256), 1, 1], [256, 1, 1], 0, args)
        }
    }

    /// The gate of the n-gram block, the gated value and its norm:
    ///
    /// ```text
    /// s        = (rms(key[c]) * norm_key[c] . rms(r[c]) * norm_query[c]) / sqrt(n)
    /// g[c]     = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
    /// gated[c] = value * g[c]
    /// nrm[c]   = rms(gated[c]) * norm_conv[c]
    /// ```
    ///
    /// `key`, `r`, `gated`, `nrm` are `[row][hc][n]`, `value` `[row][n]`, the
    /// three weights `[hc][n]`; `gate_out` (`[row][hc]`) receives `g` when given.
    ///
    /// # Safety
    ///
    /// The buffers must address these sizes (f32) on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ngram_gate(
        &self,
        stream: &Stream,
        shape: NgramShape,
        eps: f32,
        key: DevicePtr,
        value: DevicePtr,
        r: DevicePtr,
        norm_key: DevicePtr,
        norm_query: DevicePtr,
        norm_conv: DevicePtr,
        gated: DevicePtr,
        nrm: DevicePtr,
        gate_out: Option<DevicePtr>,
        n_rows: u32,
    ) -> Result<()> {
        shape.check()?;
        let blocks = n_rows
            .checked_mul(shape.n_stream)
            .ok_or_else(|| invalid("n-gram gate", format!("{n_rows} rows")))?;
        if blocks == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(key)
            .ptr(value)
            .ptr(r)
            .ptr(norm_key)
            .ptr(norm_query)
            .ptr(norm_conv)
            .ptr(gated)
            .ptr(nrm)
            .ptr(gate_out.unwrap_or(DevicePtr::NULL))
            .u32(n_rows)
            .u32(shape.n_stream)
            .u32(shape.n_embd)
            .f32(eps);
        // SAFETY: the argument list matches ares_q4_ngram_gate.
        unsafe { self.ngram_gate.launch(stream, [blocks, 1, 1], [256, 1, 1], 0, args) }
    }

    /// The dilated convolution of the n-gram block over the histories of the
    /// sequences' slots and the rows' `nrm`, and the residual update `r +=
    /// gated + silu(conv)`. The histories are not written.
    ///
    /// - `taps`: `ple_conv1d` as the file stores it, `[wide][conv_kernel]` of
    ///   `taps_type` (F16 or F32).
    /// - `history`: `[slot][back][wide]` f32, oldest first, `slot_stride`
    ///   values per slot.
    ///
    /// # Safety
    ///
    /// `r`, `gated` and `nrm` must address `n_rows * wide` f32 values and the
    /// rest the sizes above, on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ngram_window(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: NgramShape,
        r: DevicePtr,
        gated: DevicePtr,
        nrm: DevicePtr,
        taps: DevicePtr,
        taps_type: GgmlType,
        history: DevicePtr,
        slot_stride: u64,
    ) -> Result<()> {
        shape.check()?;
        let taps_f16 = match taps_type {
            GgmlType::F16 => 1,
            GgmlType::F32 => 0,
            other => return Err(invalid("n-gram convolution", format!("taps of type {other}"))),
        };
        if dims.n_segs == 0 || dims.max_seg_rows == 0 {
            return Ok(());
        }
        let chunk_rows = 32u32;
        let args = Args::new()
            .ptr(step)
            .ptr(r)
            .ptr(gated)
            .ptr(nrm)
            .ptr(taps)
            .u32(taps_f16)
            .ptr(history)
            .u64(slot_stride)
            .u32(shape.wide())
            .u32(shape.conv_kernel)
            .u32(shape.dilation)
            .u32(chunk_rows);
        // SAFETY: the argument list matches ares_q4_ngram_window.
        unsafe {
            self.ngram_window.launch(
                stream,
                [shape.wide().div_ceil(256), dims.max_seg_rows.div_ceil(chunk_rows), dims.n_segs],
                [256, 1, 1],
                0,
                args,
            )
        }
    }

    /// The commit of the n-gram history: it becomes the last `back` of (the
    /// old history, then the `nrm` of the accepted rows). `which` selects the
    /// segments as in `LayerKernels::conv_commit`.
    ///
    /// # Safety
    ///
    /// `nrm` must still hold what [`Q4Kernels::ngram_gate`] wrote for the step.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ngram_commit(
        &self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
        shape: NgramShape,
        which: CommitSet,
        nrm: DevicePtr,
        history: DevicePtr,
        slot_stride: u64,
    ) -> Result<()> {
        shape.check()?;
        if dims.n_segs == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(step)
            .ptr(nrm)
            .ptr(history)
            .u64(slot_stride)
            .u32(shape.wide())
            .u32(shape.back())
            .u32(which as u32);
        // SAFETY: the argument list matches ares_q4_ngram_commit.
        unsafe {
            self.ngram_commit.launch(
                stream,
                [shape.wide().div_ceil(256), dims.n_segs, 1],
                [256, 1, 1],
                0,
                args,
            )
        }
    }
}

/// A matrix of a stream mixer on the device, of any weight type: the kernels multiply
/// unquantized values with its weights as ggml decodes them.
#[derive(Clone, Copy, Debug)]
pub struct MixerMatrix {
    /// Where the matrix is, rows contiguous, as the file stores them.
    pub ptr: DevicePtr,
    /// Its type.
    pub ty: GgmlType,
}

fn is_float(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

impl MixerMatrix {
    /// Checks that the matrix's type holds weights and that rows of `values` values are
    /// whole blocks of it, and gives the type's ggml id, which the kernels know it by.
    fn code(&self, values: u32) -> Result<u32> {
        let ty = self.ty;
        if !ty.is_weight() || !values.is_multiple_of(ty.block_values()) {
            return Err(invalid(
                "read",
                format!("a mixer matrix of {ty} in rows of {values} values (whole blocks)"),
            ));
        }
        Ok(ty.id())
    }
}

/// The two launches of a fused read for matrices of one type, and the most threads a block
/// of each can have on the device (what its registers leave).
#[derive(Clone, Debug)]
struct Read {
    ty: GgmlType,
    down: Function,
    up: Function,
    max_threads: u32,
    max_threads_up: u32,
}

impl Read {
    /// The pair `ares_q4_read_down{suffix}`, `ares_q4_read_up{suffix}` of `module`.
    fn load(module: &Module, suffix: &str) -> Result<Self> {
        let down = module.function(&format!("ares_q4_read_down{suffix}"))?;
        let up = module.function(&format!("ares_q4_read_up{suffix}"))?;
        Ok(Self {
            ty: GgmlType::F32,
            max_threads: down.max_threads()?,
            max_threads_up: up.max_threads()?,
            down,
            up,
        })
    }
}

/// What the write in front of a read adds to the streams
/// ([`Q4Kernels::read_down`]).
#[derive(Clone, Copy, Debug)]
pub struct ReadWrite {
    /// This device's output of the last half block, `rows * n` f32, or `None`
    /// when only the peer has one.
    pub own: Option<DevicePtr>,
    /// The inject values of the read before, `rows * hc` f32.
    pub inj: DevicePtr,
    /// The peer's partial output of the same half block, which the kernel
    /// waits for.
    pub peer: Option<Arrival>,
}

/// Something a peer device writes into this device's memory, and the flags
/// that announce it.
#[derive(Clone, Copy, Debug)]
pub struct Arrival {
    /// Where the data arrives.
    pub data: DevicePtr,
    /// The flags of the exchange (`u32`), in this device's memory.
    pub flags: DevicePtr,
    /// The number of the exchange.
    pub token: u32,
}

/// Where a kernel writes into the peer's memory, and the flags it sets there
/// when the data is written.
#[derive(Clone, Copy, Debug)]
pub struct Departure {
    /// Where the data goes (peer memory: write only).
    pub data: DevicePtr,
    /// The peer's flags (peer memory: write only).
    pub flags: DevicePtr,
    /// The number of the exchange.
    pub token: u32,
}

/// The quantized types whose fused read has its decoder inlined (`kernels/q4_reads.cu`):
/// those files use for the stream mixers. Every other type is read by the kernels that
/// decode by the type at run time.
const INLINED_READS: [GgmlType; 9] = [
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::IQ4_NL,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
];

/// The most rows of `down` one block of the read's first launch computes
/// (`ARES_Q4_TILE`).
const READ_TILE: u32 = 32;
/// The most residual streams and the largest rank the fused read takes
/// (`ARES_Q4_MAX_STREAMS`, `ARES_Q4_MAX_RANK`).
const READ_MAX_STREAMS: u32 = 8;
const READ_MAX_RANK: u32 = 1024;

/// Threads of a block that keeps a row of `n` values in registers, a float4
/// per thread and resident chunk (`ARES_Q4_RESIDENT` is 2).
fn row_block(what: &'static str, n: u32) -> Result<u32> {
    let chunks = n / 4;
    let threads = if chunks <= 1024 { chunks } else { chunks.div_ceil(2) };
    let threads = threads.next_multiple_of(32).max(64);
    if n == 0 || !n.is_multiple_of(32) || threads > 1024 {
        return Err(invalid(what, format!("rows of {n} values (a multiple of 32, at most 8192)")));
    }
    Ok(threads)
}

impl Q4Kernels {
    /// Whether [`Q4Kernels::read_down`] and [`Q4Kernels::read_up`] can run a
    /// mixer of these sizes and types; otherwise a read takes the separate
    /// kernels.
    /// Every weight type is read: `down` and `inject` in rows of `hc * n` values, `up`
    /// in rows of `rank`, as the file stores them.
    pub fn reads_fused(hc: u32, n: u32, rank: u32, matrices: &[GgmlType]) -> bool {
        (1..=READ_MAX_STREAMS).contains(&hc)
            && row_block("mixer", n).is_ok()
            && rank > 0
            && rank <= READ_MAX_RANK
            && rank.is_multiple_of(4)
            && matrices.iter().all(|ty| ty.is_weight())
    }

    /// The f32 values of the scratch between the two launches of a fused
    /// read: the partial sums per stream of `down @ xn` and `inject @ xn`.
    pub fn read_partials(rows: u32, hc: u32, rank: u32) -> u64 {
        u64::from(rows) * u64::from(hc) * u64::from(rank + hc)
    }

    /// The flags [`Q4Kernels::read_up`] sets for `rows` rows of `n` values.
    pub fn read_flags(rows: u32, n: u32) -> u32 {
        rows * (n / 32)
    }

    /// The first launch of a fused read, with the write of the last half
    /// block's output in front of it when there is one:
    ///
    /// - `write`: `r_dst[row][c] = r_src[row][c] + (own[row] + peer[row]) * 2
    ///   sigmoid(inj[row][c] / hc)`; without a write `r_dst` is not written.
    /// - `norm` (`(weights, xn, down, inject)`): `xn[row][c]` is
    ///   `rms(r[row][c])` times `weights[c]`, and per stream the partial sums
    ///   of `down @ xn` and of
    ///   `inject @ xn` into `partials` ([`Q4Kernels::read_partials`] f32),
    ///   which [`Q4Kernels::read_up`] adds up. Without `norm` the launch is
    ///   the write alone.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `r_src` and
    /// `r_dst` `rows * hc * n` f32 (the same buffer only without a norm or
    /// without a write), `xn` the same, the weights `hc * n` f32, `down`
    /// `rank` rows and `inject` `hc` rows of `hc * n` values of their types,
    /// everything 16-byte aligned; a peer's data `rows * n` f32 and `rows`
    /// flags that a kernel of the peer sets to the token.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn read_down(
        &self,
        stream: &Stream,
        r_src: DevicePtr,
        r_dst: DevicePtr,
        write: Option<ReadWrite>,
        norm: Option<(DevicePtr, DevicePtr, MixerMatrix, Option<MixerMatrix>)>,
        partials: DevicePtr,
        rows: u32,
        hc: u32,
        n: u32,
        rank: u32,
        eps: f32,
    ) -> Result<()> {
        let threads = row_block("read", n)?;
        if hc == 0 || hc > READ_MAX_STREAMS || rows == 0 {
            return Err(invalid("read", format!("{rows} rows of {hc} streams")));
        }
        let null = DevicePtr::NULL;
        let (own, inj, peer) = match write {
            Some(w) => (w.own.unwrap_or(null), w.inj, w.peer),
            None => (null, null, None),
        };
        let (peer_out, peer_flags, token) =
            peer.map_or((null, null, 0), |p| (p.data, p.flags, p.token));
        let mut function = &self.reads_of(GgmlType::F32)?.down;
        let (weights, xn, down, inject, types, tiles) = match norm {
            Some((weights, xn, down, inject)) => {
                let down_ty = down.code(hc * n)?;
                let inject_ty = inject.map(|m| m.code(hc * n)).transpose()?;
                // The read of `down`'s type takes a float `inject` and blocks as wide as its
                // registers allow; any other takes the read that decodes everything by its type.
                let read = self.reads_of(down.ty)?;
                let float_inject = inject.is_none_or(|m| is_float(m.ty));
                function = if float_inject && threads <= read.max_threads {
                    &read.down
                } else {
                    &self.read_any.down
                };
                if rank == 0 || rank > READ_MAX_RANK {
                    return Err(invalid("read", format!("a mixer of rank {rank}")));
                }
                // The fewest tiles of `down` the kernel's 32 rows allow: a
                // block has a fixed cost (the norm of its stream, the sums
                // of its rows over the block), and more blocks than that
                // made a read slower on both cards measured (27 to 45 us and
                // 80 to 91 us with every multiprocessor filled).
                let tiles = rank.div_ceil(READ_TILE);
                (
                    weights,
                    xn,
                    down.ptr,
                    inject.map_or(null, |m| m.ptr),
                    down_ty | (inject_ty.unwrap_or(0) << 8),
                    tiles,
                )
            }
            None => (null, null, null, null, 0, 1),
        };
        check_align("read", &[r_src, r_dst, own, peer_out, weights, xn, down, inject], 16)?;
        let args = Args::new()
            .ptr(r_src)
            .ptr(r_dst)
            .ptr(own)
            .ptr(peer_out)
            .ptr(peer_flags)
            .u32(token)
            .ptr(inj)
            .ptr(weights)
            .ptr(xn)
            .ptr(down)
            .ptr(inject)
            .ptr(partials)
            .u32(types | (hc << 16) | (tiles << 24))
            .u32(n)
            .u32(rank)
            .f32(eps);
        // SAFETY: the argument list matches the ares_q4_read_down kernels (one per type of
        // `down`); the ranges are the caller's guarantee.
        unsafe { function.launch(stream, [hc * tiles, rows, 1], [threads, 1, 1], 0, args) }
    }

    /// The second launch of a fused read: `lo = silu(down / hc)`, `gate = up
    /// @ lo`, the gated mean of the streams, written as quantized activations
    /// over `rows * n` values into `act` (the layout of
    /// `Kernels::quantize_activations`), and the inject values of the read
    /// into `inj` (`rows * hc` f32) when it is given. With `peer` the
    /// activations are written into the peer's memory as well and
    /// [`Q4Kernels::read_flags`] flags are set there.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `xn`,
    /// and `partials` as [`Q4Kernels::read_down`] wrote them, `up`
    /// `hc * n` rows of `rank` values, `act` the bytes of the layout, 16-byte
    /// aligned. A peer's data and flags must be memory of a device this one
    /// has peer access to, of the same sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn read_up(
        &self,
        stream: &Stream,
        xn: DevicePtr,
        partials: DevicePtr,
        up: MixerMatrix,
        act: DevicePtr,
        inj: Option<DevicePtr>,
        peer: Option<Departure>,
        rows: u32,
        hc: u32,
        n: u32,
        rank: u32,
    ) -> Result<()> {
        let up_ty = up.code(rank)?;
        if !Self::reads_fused(hc, n, rank, &[]) || rows == 0 {
            return Err(invalid("read", format!("{rows} rows, {hc} streams of {n}, rank {rank}")));
        }
        // As many threads per row of `up` as divide the rank, in blocks of
        // up to 512 threads: more threads per row shorten the chain of reads
        // each makes, more threads per block than that put fewer blocks on a
        // multiprocessor at once.
        let mut tpr = 1u32;
        while tpr < 32 && 64 * hc * tpr <= 512 && rank.is_multiple_of(8 * tpr) {
            tpr *= 2;
        }
        let null = DevicePtr::NULL;
        let (peer_act, peer_flags, token) =
            peer.map_or((null, null, 0), |p| (p.data, p.flags, p.token));
        check_align("read", &[xn, up.ptr, act, peer_act], 16)?;
        let args = Args::new()
            .ptr(xn)
            .ptr(partials)
            .ptr(up.ptr)
            .ptr(act)
            .ptr(inj.unwrap_or(null))
            .ptr(peer_act)
            .ptr(peer_flags)
            .u32(token)
            .u32(up_ty)
            .u32(hc)
            .u32(n)
            .u32(rank)
            .u32(tpr);
        let read = self.reads_of(up.ty)?;
        let fits = 32 * hc * tpr <= read.max_threads_up;
        let function = if fits { &read.up } else { &self.read_any.up };
        // SAFETY: the argument list matches the ares_q4_read_up kernels (one per type of
        // `up`).
        unsafe { function.launch(stream, [n / 32, rows, 1], [32 * hc * tpr, 1, 1], 0, args) }
    }

    /// The worker's side of an exchange between two devices, either half or
    /// both, in one launch:
    ///
    /// - `push` (`(a, b, s, to)`): writes `a[row] + b[row] * sigmoid(s[row])`
    ///   (`rows * n` f32; a missing `a` or `b` is zero, a missing `s` a factor
    ///   of 1) into the peer's memory and sets a flag per row there.
    /// - `wait` (`(flags, n_flags, token)`): then waits until that many of
    ///   this device's flags hold the token.
    /// - `gate` (`(inp, m, s_out)`): then `s_out[row] = inp . m[row]`, the
    ///   scalar gate of the shared expert from the mixed input that arrived.
    /// - `zero`: then clears `rows * n` f32.
    ///
    /// # Safety
    ///
    /// Every pointer on the stream's device (the departure in memory of a
    /// device it has peer access to), of the sizes above, 16-byte aligned,
    /// alive until the launch has run.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub unsafe fn swap(
        &self,
        stream: &Stream,
        push: Option<(Option<DevicePtr>, Option<DevicePtr>, Option<DevicePtr>, Departure)>,
        wait: Option<(DevicePtr, u32, u32)>,
        gate: Option<(DevicePtr, DevicePtr, DevicePtr)>,
        zero: Option<DevicePtr>,
        rows: u32,
        n: u32,
    ) -> Result<()> {
        let threads = row_block("swap", n)?;
        if rows == 0 || (push.is_none() && wait.is_none()) {
            return Ok(());
        }
        let null = DevicePtr::NULL;
        let (a, b, s, to) = match push {
            Some((a, b, s, to)) => {
                (a.unwrap_or(null), b.unwrap_or(null), s.unwrap_or(null), Some(to))
            }
            None => (null, null, null, None),
        };
        let (peer_dst, peer_flags, token_out) =
            to.map_or((null, null, 0), |d| (d.data, d.flags, d.token));
        let (own_flags, n_flags, token_in) = wait.unwrap_or((null, 0, 0));
        let (inp, m, s_out) = gate.unwrap_or((null, null, null));
        check_align("swap", &[a, b, peer_dst, inp, m, zero.unwrap_or(null)], 16)?;
        let args = Args::new()
            .ptr(a)
            .ptr(b)
            .ptr(s)
            .ptr(peer_dst)
            .ptr(peer_flags)
            .u32(token_out)
            .ptr(own_flags)
            .u32(n_flags)
            .u32(token_in)
            .ptr(inp)
            .ptr(m)
            .ptr(s_out)
            .ptr(zero.unwrap_or(null))
            .u32(rows)
            .u32(n);
        // SAFETY: the argument list matches ares_q4_swap.
        unsafe { self.swap.launch(stream, [rows, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// Cuts the route record of a layer into this device's record (`own`) and
    /// its peer's, which is written into the peer's memory and announced by
    /// one flag: the groups of experts whose byte in `owner` (one per expert)
    /// is not 0 go to the peer. `zero`, when given, is cleared first
    /// (`(buffer, f32 values)`): the buffer this device's experts add to.
    ///
    /// # Safety
    ///
    /// `record` and `own` must be records for `cap` pairs (`rows * top_k`)
    /// on the stream's device, `owner` a byte per expert of the layer, the
    /// departure a record of the same size in memory of a device this one
    /// has peer access to, and its flag.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn route_split(
        &self,
        stream: &Stream,
        record: DevicePtr,
        owner: DevicePtr,
        own: DevicePtr,
        peer: Departure,
        zero: Option<(DevicePtr, u32)>,
        cap: u32,
    ) -> Result<()> {
        let (zero, zero_values) = zero.unwrap_or((DevicePtr::NULL, 0));
        check_align("route split", &[record, own, peer.data, zero], 16)?;
        let args = Args::new()
            .ptr(record)
            .ptr(owner)
            .ptr(own)
            .ptr(peer.data)
            .ptr(peer.flags)
            .u32(peer.token)
            .ptr(zero)
            .u32(zero_values)
            .u32(cap);
        // SAFETY: the argument list matches ares_q4_route_split.
        unsafe { self.route_split.launch(stream, [1, 1, 1], [256, 1, 1], 0, args) }
    }

    /// Waits for the record the peer's [`Q4Kernels::route_split`] writes into
    /// `arrival.data`, then gives every group its slot from `slot_table`
    /// (this device's table of the layer) and counts the missing ones.
    ///
    /// # Safety
    ///
    /// `arrival.data` must be a record for `cap` pairs on the stream's
    /// device, `slot_table` an `i32` per expert, `arrival.flags` one flag.
    pub unsafe fn route_recv(
        &self,
        stream: &Stream,
        arrival: Arrival,
        slot_table: DevicePtr,
        cap: u32,
    ) -> Result<()> {
        check_align("route record", &[arrival.data, slot_table], 4)?;
        let args = Args::new()
            .ptr(arrival.data)
            .ptr(slot_table)
            .ptr(arrival.flags)
            .u32(arrival.token)
            .u32(cap);
        // SAFETY: the argument list matches ares_q4_route_recv.
        unsafe { self.route_recv.launch(stream, [1, 1, 1], [32, 1, 1], 0, args) }
    }
}
