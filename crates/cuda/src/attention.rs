//! Attention on the device: writing K and V into the cache in the engine's own
//! formats, and attention that reads them natively.
//!
//! The kernels are in `kernels/attention.cu` (a module of its own, compute
//! capability 8.0 and newer, like the prefill kernels) and the formats they
//! write are the ones of `ares_quant::kv`, byte for byte. `ares_quant::kvhead`
//! is the reference both are held to.
//!
//! # The stored sequence on the device
//!
//! A sequence is a **table of entries**, one `u32` per [`ENTRY_CELLS`] (128)
//! cells, which is one tile of the formats with a mean. Bit 31 of an entry says
//! which pool it is in (set: the converted pool), the low 31 bits are the slot
//! in that pool. The table is the same for every attention layer; each layer
//! and device has its own pair of pools, indexed by the slot, holding only the
//! KV heads of that device.
//!
//! - An **exact** slot holds, for each KV head, 128 rows of K and 128 rows of V
//!   as f16 (`2 * D` bytes per row, K first). [`CacheGeometry::exact_entry_bytes`].
//! - A **converted** slot holds, for each KV head, `[K mean][K rows][V mean][V
//!   rows]`: 128 rows in the format of the policy (`format.row_bytes(D)` each,
//!   the bytes `KvHead` stores) and, for q7 to q4, the mean of the tile's rotated
//!   rows as `D` f16 before them. [`CacheGeometry::converted_entry_bytes`]. All
//!   sections start at a multiple of 16 bytes (rows of `D = 128` are 8-byte
//!   aligned; a group is 2 + 4 * bits bytes, so codes are 2-byte aligned).
//!
//! With a mean in K or V (q7 to q4), the first tile and the newest tiles of a
//! sequence are exact entries, and a tile becomes a converted entry
//! ([`AttentionKernels::convert`]) when the policy says so. Without one
//! (f16 and q8), cells are written straight into the format in converted
//! entries, the entry that is still filling included. A draft row of such a
//! policy is encoded from its f32 values like any other, whereas the reference
//! keeps a pending cell as f16 until it is settled; the stored bytes of an
//! accepted draft then differ by that one f16 rounding of the input, and rejected
//! drafts are forgotten by the table shrinking.
//!
//! The pools are one array of slots per attention layer and device. A pair of
//! pools costs, for `D = 256` and 4 KV heads, 512 KiB per exact slot and 136 KiB
//! (q8/q8) to 74 KiB (q4/q4) per converted slot, 128 cells each. No scratch is
//! reserved per request; a step's scratch is [`Plan::state_bytes`]
//! (`state_slots * q_heads * (head_dim + 1) * 4` bytes, tens of KiB for a decode
//! step with one row, a few MiB for eight rows of four sequences against long
//! histories) and the plan, a few KiB.
//!
//! # What `ares-state` has to provide
//!
//! - A page is one entry (`cells_per_page = 128`). Its table is the `u32` array
//!   above; `Sequence::pages()` gives the pages in order and a map from page to
//!   (pool, slot) turns them into entries. `Sequence::take_dirty` says which entries
//!   of the table changed since the last upload.
//! - The map is **per page, not per sequence**: converting a shared page converts it for
//!   every holder. A page that becomes a converted entry moves from the exact pool to the
//!   converted pool (a new slot, the exact one is freed when the conversion has run on the
//!   stream), so the state manager owns two slot allocators of different slot sizes and the
//!   ledger has two kinds of unit instead of one `bytes_per_page`: the exact pool is sized
//!   by the most sequences that run at once times `1 + ceil(newest / 128) + 1` entries (and
//!   the draft window), the converted pool by what is left. A cached prefix keeps its exact
//!   tail entries in the exact pool; under pressure the cache may convert them (irreversible,
//!   a little less precise) to give the slots back.
//! - The conversion is decided by the sequence length the way `KvPolicy::quantized_cells`
//!   says, after the accepted rows of a step are known (`settle`); drafts are exact entries
//!   until then. The state manager emits (exact slot, converted slot) pairs per step, which
//!   [`AttentionKernels::convert`] takes for every layer.
//! - Tables of a step go to the device with the step record; the kernels only read them.
//!
//! # Reading
//!
//! [`Planner`] cuts a step into **items**, one thread block each, and
//! [`AttentionKernels::attend`] launches one kernel per shape. An item is some rows
//! of one sequence (their query heads of one KV head are the "vectors" of the
//! product) and a list of ranges of cells, in entries of one kind. Rows of a decode window
//! share the K and V they read; dense causal attention is one range per item, block-sparse
//! attention (qwen4exp) one range per selected block. Items that are only part of a row's
//! cells write a state (the log2 of the sum of exponentials, the weighted V in the
//! original domain) and [`attend`](AttentionKernels::attend) merges them, then the
//! sigmoid gate is applied as the epilogue.
//!
//! In a block, K and V of a chunk of 32 cells (16 for the wide kernel) are copied raw
//! from global memory by `cp.async`, decoded to f16 tiles in shared memory (the
//! rotated domain for the `qB` formats; the tile mean is not part of the tile and
//! enters the scores and the accumulator once per tile) and multiplied on the tensor
//! cores with f32 accumulation. A decode step has few vectors, so the cells are the
//! rows of the product (16 of them) and the vectors the columns (tiles of 8); a wide
//! step has the vectors as rows. The queries are rotated once per block.
//!
//! **Error.** The inputs of the tensor cores are f16: the query scaled by `scale * log2(e)`,
//! the decoded K and V (code times scale rounded once to f16) and the softmax weights.
//! Against `KvHead::attend` on the same stored bytes that is about 1e-3 of the largest
//! output element at worst in the tests (5e-4 to 3e-3 on random rows and on the captured
//! rows of Qwen3.8-27B, 2e-4 to 3e-3), and it does not depend on the format: the
//! formats' own errors are in the stored bytes, which the kernel reads exactly. The
//! softmax, the sums and the merges are f32.

mod plan;

pub use plan::{AttnItem, AttnShape, Plan, PlanConfig, Planner};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.fatbin"));

/// Cells per table entry: one tile.
pub const ENTRY_CELLS: u32 = 128;

/// Bits of the status word of the writing kernels.
pub const ATTN_STATUS_NONFINITE: u32 = 1;
/// The rotation of a group overflowed f32.
pub const ATTN_STATUS_OVERFLOW: u32 = 2;
/// A row was written directly into an entry of a format with a mean.
pub const ATTN_STATUS_DIRECT_MEAN: u32 = 4;

/// A storage format of K or V rows: the ladder of `ares_quant::KvFormat`, in the
/// same order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheFormat {
    F16,
    Q8,
    Q7,
    Q6,
    Q5,
    Q4,
}

impl CacheFormat {
    pub const ALL: [CacheFormat; 6] = [
        CacheFormat::F16,
        CacheFormat::Q8,
        CacheFormat::Q7,
        CacheFormat::Q6,
        CacheFormat::Q5,
        CacheFormat::Q4,
    ];

    /// The number the kernels use (the index in [`CacheFormat::ALL`]).
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// Code width of a `qB` format, 0 for f16.
    pub const fn bits(self) -> u32 {
        match self {
            CacheFormat::F16 => 0,
            other => 9 - other as u32,
        }
    }

    /// Bytes of a group of 32 values.
    pub const fn group_bytes(self) -> u32 {
        match self {
            CacheFormat::F16 => 64,
            other => 2 + 4 * other.bits(),
        }
    }

    /// Whether rows are stored per tile relative to a mean (q7 to q4).
    pub const fn has_mean(self) -> bool {
        self as u32 >= 2
    }

    /// Whether stored values are rotated (every `qB` format).
    pub const fn is_rotated(self) -> bool {
        self as u32 >= 1
    }

    /// Bytes of a row of `head_dim` values.
    pub const fn row_bytes(self, head_dim: u32) -> u32 {
        head_dim / 32 * self.group_bytes()
    }

    /// Bytes of the mean of a tile (`head_dim` f16), 0 without one.
    pub const fn mean_bytes(self, head_dim: u32) -> u32 {
        if self.has_mean() { 2 * head_dim } else { 0 }
    }
}

/// The formats and head counts one set of pools is laid out for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheGeometry {
    pub k: CacheFormat,
    pub v: CacheFormat,
    /// Values per head: 128 or 256 (a multiple of 32).
    pub head_dim: u32,
    /// KV heads held in the pools of this device.
    pub kv_heads: u32,
    /// KV heads of the model (the stride of the K and V rows a step writes).
    pub kv_total: u32,
    /// The first KV head of the model that this device holds.
    pub first_kv_head: u32,
    /// Query heads per KV head.
    pub q_per_kv: u32,
}

impl CacheGeometry {
    /// Bytes of an exact slot: `kv_heads * (K + V rows of 128 cells as f16)`.
    pub const fn exact_entry_bytes(&self) -> u64 {
        self.kv_heads as u64 * ENTRY_CELLS as u64 * 4 * self.head_dim as u64
    }

    /// Bytes of one head in a converted slot.
    pub const fn converted_head_bytes(&self) -> u64 {
        self.k.mean_bytes(self.head_dim) as u64
            + self.v.mean_bytes(self.head_dim) as u64
            + ENTRY_CELLS as u64
                * (self.k.row_bytes(self.head_dim) + self.v.row_bytes(self.head_dim)) as u64
    }

    /// Bytes of a converted slot.
    pub const fn converted_entry_bytes(&self) -> u64 {
        self.kv_heads as u64 * self.converted_head_bytes()
    }

    /// The 64-bit configuration word the kernels take.
    pub fn config(&self) -> u64 {
        u64::from(self.k.id())
            | u64::from(self.v.id()) << 4
            | u64::from(self.kv_heads) << 8
            | u64::from(self.kv_total) << 16
            | u64::from(self.first_kv_head) << 24
            | u64::from(self.q_per_kv) << 32
            | u64::from(self.head_dim) << 40
    }

    fn check(&self) -> Result<()> {
        let bad = |detail: String| Err(Error::Invalid { what: "cache geometry", detail });
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(32) || self.head_dim > 4095 {
            return bad(format!("head size {} is not a multiple of 32", self.head_dim));
        }
        if self.kv_heads == 0
            || self.kv_heads > 255
            || self.kv_total > 255
            || self.first_kv_head + self.kv_heads > self.kv_total
            || self.q_per_kv == 0
            || self.q_per_kv > 255
        {
            return bad(format!("head counts {self:?}"));
        }
        Ok(())
    }
}

/// One row of a step as the kernels see it (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttnRow {
    /// Index of the first entry of the row's sequence in the table array.
    pub table: u32,
    /// Position of the cell the row writes; a causal row sees cells `0..=pos`.
    pub pos: u32,
    /// First state slot of the row (see the attention kernels).
    pub slot_base: u32,
    /// Number of pieces the row has.
    pub n_slots: u32,
}

impl AttnRow {
    /// The rows as the bytes to upload.
    pub fn to_bytes(rows: &[AttnRow]) -> Vec<u8> {
        rows.iter()
            .flat_map(|r| [r.table, r.pos, r.slot_base, r.n_slots])
            .flat_map(u32::to_le_bytes)
            .collect()
    }
}

/// The attention module on one device.
#[derive(Clone, Debug)]
pub struct AttentionKernels {
    device: Device,
    append: Function,
    convert: Function,
    encode_tile_f32: Function,
    /// The attention kernels by head size and shape.
    attend: Vec<(u32, AttnShape, Function)>,
    merge: Vec<(u32, Function)>,
}

/// A plan on the device.
#[derive(Debug)]
pub struct PlanBuffers {
    items: DeviceBuffer,
    ranges: DeviceBuffer,
    rows: DeviceBuffer,
    row_ids: DeviceBuffer,
    groups: Vec<(AttnShape, usize, usize)>,
    n_rows: u32,
    state_slots: u64,
    needs_merge: bool,
}

impl PlanBuffers {
    /// The rows of the plan on the device (for the append kernel).
    pub fn rows(&self) -> DevicePtr {
        self.rows.ptr()
    }
}

impl Plan {
    /// Uploads the plan (synchronously; the step loop writes the same bytes into
    /// its step record).
    pub fn upload(&self, device: &Device) -> Result<PlanBuffers> {
        let put = |bytes: Vec<u8>| -> Result<DeviceBuffer> {
            let buffer = DeviceBuffer::new(device, bytes.len().max(16) as u64)?;
            buffer.write(0, &bytes)?;
            Ok(buffer)
        };
        let words = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        Ok(PlanBuffers {
            items: put(AttnItem::to_bytes(&self.items))?,
            ranges: put(words(&self.ranges))?,
            rows: put(AttnRow::to_bytes(&self.rows))?,
            row_ids: put(words(&self.row_ids))?,
            groups: self.groups.iter().map(|(s, r)| (*s, r.start, r.end - r.start)).collect(),
            n_rows: self.rows.len() as u32,
            state_slots: u64::from(self.state_slots),
            needs_merge: self.needs_merge,
        })
    }
}

/// The buffers an attention launch reads and writes.
#[derive(Clone, Copy, Debug)]
pub struct AttendBuffers {
    /// The u32 entry array the plan's tables index.
    pub tables: DevicePtr,
    pub exact_pool: DevicePtr,
    pub converted_pool: DevicePtr,
    /// `[row][q_heads_of_model][head_dim]` f32, scaled by nothing (the kernel applies `scale`).
    pub q: DevicePtr,
    /// The gate (same layout as `q`), applied as `out * sigmoid(gate)` when present.
    pub gate: Option<DevicePtr>,
    /// `[row][q_heads_of_model][head_dim]` f32; the rows and heads of this plan are written.
    pub out: DevicePtr,
    /// Scratch for partial states: [`Plan::state_bytes`] bytes.
    pub scratch: DevicePtr,
}

impl AttentionKernels {
    /// Loads the module. `ARES_ATTENTION_FATBIN` names a file to load instead of the
    /// embedded image (for kernel development: no rebuild of the crate).
    pub fn load(device: &Device) -> Result<Self> {
        let module = match std::env::var_os("ARES_ATTENTION_FATBIN") {
            Some(path) => {
                let image = std::fs::read(&path).map_err(|e| Error::Invalid {
                    what: "ARES_ATTENTION_FATBIN",
                    detail: format!("{}: {e}", path.to_string_lossy()),
                })?;
                Module::load(device, &image)?
            }
            None => Module::load(device, IMAGE)?,
        };
        let mut attend = Vec::new();
        let mut merge = Vec::new();
        for d in [256u32, 128] {
            for shape in AttnShape::ALL {
                let f = module.function(&shape.kernel_name(d))?;
                f.allow_dynamic_shared(shape.smem_bytes(d))?;
                attend.push((d, shape, f));
            }
            merge.push((d, module.function(&format!("ares_attn_merge_d{d}"))?));
        }
        Ok(Self {
            device: device.clone(),
            append: module.function("ares_attn_append")?,
            convert: module.function("ares_attn_convert")?,
            encode_tile_f32: module.function("ares_attn_encode_tile_f32")?,
            attend,
            merge,
        })
    }

    /// Attention of the rows of a plan: one launch per kernel shape the plan uses, then
    /// the merge of partial states if any row has more than one piece. `scale` is
    /// the factor on the scores (`1 / sqrt(head_dim)`).
    ///
    /// # Safety
    ///
    /// The buffers must be live, on the stream's device and of the sizes the plan and
    /// the geometry imply; the tables must hold the entries the plan names.
    pub unsafe fn attend(
        &self,
        stream: &Stream,
        geo: &CacheGeometry,
        plan: &PlanBuffers,
        buffers: &AttendBuffers,
        scale: f32,
    ) -> Result<()> {
        geo.check()?;
        let d = geo.head_dim;
        let hq = u64::from(geo.kv_heads * geo.q_per_kv);
        let acc_bytes = plan_state_floats(plan, hq, d) * 4;
        let st_acc = buffers.scratch;
        let st_lse = buffers.scratch.offset(acc_bytes);
        let gate = buffers.gate.unwrap_or(DevicePtr::NULL);
        for (shape, first, count) in &plan.groups {
            let function = self
                .attend
                .iter()
                .find(|(dd, s, _)| *dd == d && s == shape)
                .map(|(_, _, f)| f)
                .ok_or_else(|| Error::Invalid {
                    what: "attention",
                    detail: format!("no kernel for a head size of {d}"),
                })?;
            let args = Args::new()
                .ptr(plan.items.ptr().offset(*first as u64 * 48))
                .ptr(plan.ranges.ptr())
                .ptr(plan.rows.ptr())
                .ptr(plan.row_ids.ptr())
                .ptr(buffers.tables)
                .ptr(buffers.exact_pool)
                .ptr(buffers.converted_pool)
                .ptr(buffers.q)
                .ptr(gate)
                .ptr(buffers.out)
                .ptr(st_acc)
                .ptr(st_lse)
                .u64(geo.config())
                .f32(scale * std::f32::consts::LOG2_E)
                .u32(shape.rawcap(d));
            // SAFETY: the caller's contract.
            unsafe {
                function.launch(
                    stream,
                    [*count as u32, 1, 1],
                    [shape.threads(), 1, 1],
                    shape.smem_bytes(d),
                    args,
                )?;
            }
        }
        if plan.needs_merge {
            let function =
                self.merge.iter().find(|(dd, _)| *dd == d).map(|(_, f)| f).ok_or_else(|| {
                    Error::Invalid {
                        what: "attention",
                        detail: format!("no kernel for a head size of {d}"),
                    }
                })?;
            let args = Args::new()
                .ptr(plan.rows.ptr())
                .ptr(st_acc)
                .ptr(st_lse)
                .ptr(gate)
                .ptr(buffers.out)
                .u64(geo.config());
            // SAFETY: the caller's contract.
            unsafe {
                function.launch(
                    stream,
                    [plan.n_rows, geo.kv_heads * geo.q_per_kv, 1],
                    [d.min(256), 1, 1],
                    0,
                    args,
                )?;
            }
        }
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Appends the K and V rows of a step. `rows[i]` says where row `i` goes;
    /// `k` and `v` are `[row][kv_total][head_dim]` f32; `tables` is the u32
    /// entry array the rows' `table` fields index; `status` is one u32 the
    /// kernel or-s the `ATTN_STATUS_*` bits into.
    ///
    /// # Safety
    ///
    /// Every pointer must address live device memory of the sizes the shapes
    /// imply, on the stream's device, and the entries the rows name must exist.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn append(
        &self,
        stream: &Stream,
        geo: &CacheGeometry,
        rows: DevicePtr,
        n_rows: u32,
        k: DevicePtr,
        v: DevicePtr,
        tables: DevicePtr,
        exact_pool: DevicePtr,
        converted_pool: DevicePtr,
        status: DevicePtr,
    ) -> Result<()> {
        geo.check()?;
        if n_rows == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(rows)
            .ptr(k)
            .ptr(v)
            .ptr(tables)
            .ptr(exact_pool)
            .ptr(converted_pool)
            .u64(geo.config())
            .ptr(status);
        // SAFETY: the caller's contract.
        unsafe { self.append.launch(stream, [n_rows, geo.kv_heads, 1], [64, 1, 1], 0, args) }
    }

    /// Converts exact slots into converted ones: `jobs` is `n_jobs` pairs of u32
    /// (exact slot, converted slot).
    ///
    /// # Safety
    ///
    /// As for [`AttentionKernels::append`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn convert(
        &self,
        stream: &Stream,
        geo: &CacheGeometry,
        jobs: DevicePtr,
        n_jobs: u32,
        exact_pool: DevicePtr,
        converted_pool: DevicePtr,
        status: DevicePtr,
    ) -> Result<()> {
        geo.check()?;
        if n_jobs == 0 {
            return Ok(());
        }
        let args =
            Args::new().ptr(jobs).ptr(exact_pool).ptr(converted_pool).u64(geo.config()).ptr(status);
        // SAFETY: the caller's contract.
        unsafe {
            self.convert.launch(
                stream,
                [n_jobs, geo.kv_heads, 2],
                [256, 1, 1],
                geo.head_dim * 4,
                args,
            )
        }
    }

    /// The codec alone on f32 rows, for tests: `rows` rows of `head_dim` values
    /// as one tile (`use_mean`) or each on its own. Writes the mean
    /// (`format.mean_bytes`) and the rows.
    ///
    /// # Safety
    ///
    /// As for [`AttentionKernels::append`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn encode_tile_f32(
        &self,
        stream: &Stream,
        src: DevicePtr,
        rows: u32,
        head_dim: u32,
        format: CacheFormat,
        use_mean: bool,
        mean_out: DevicePtr,
        rows_out: DevicePtr,
        status: DevicePtr,
    ) -> Result<()> {
        let args = Args::new()
            .ptr(src)
            .u32(rows)
            .u32(head_dim)
            .u32(format.id())
            .u32(u32::from(use_mean))
            .ptr(mean_out)
            .ptr(rows_out)
            .ptr(status);
        // SAFETY: the caller's contract.
        unsafe { self.encode_tile_f32.launch(stream, [1, 1, 1], [256, 1, 1], head_dim * 4, args) }
    }
}

/// Floats of partial outputs in the scratch (the log sums follow).
fn plan_state_floats(plan: &PlanBuffers, hq: u64, d: u32) -> u64 {
    plan.state_slots * hq * u64::from(d)
}
