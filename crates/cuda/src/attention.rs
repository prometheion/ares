//! Attention on the device: writing K and V into the cache in the engine's own
//! formats, and attention that reads them natively.
//!
//! The kernels are in `kernels/attention.cu` (a module of its own, on the
//! tensor cores like the prefill kernels) and the formats they write are the
//! ones of `ares_quant::kv`, byte for byte. `ares_quant::kvhead`
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
//! A converted entry whose bit 30 is set ([`FAR_ENTRY`]) lies in the **far pool**:
//! the same converted slots in page-locked host memory that the device reads and
//! writes over its link, for the K/V of a context beyond VRAM. Bits 0 to 29 are the
//! slot there. The kernels take the far pool of a layer next to its converted pool
//! (null when nothing is far) and read the same bytes from either.
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
//! [`AttentionKernels::attend`] launches one kernel per shape and kind of entry. An item
//! is some rows (their query heads of one KV head are the "vectors" of the product) and a
//! list of ranges of cells, in entries of one kind. Rows of a decode window share the K and
//! V they read, and so do the rows of several sequences over a prefix they have in common;
//! dense causal attention is one range per item, block-sparse attention (qwen4exp) one range
//! per selected block. Items that are only part of a row's cells write a state (the log2 of
//! the sum of exponentials, the weighted V in the original domain) and
//! [`attend`](AttentionKernels::attend) merges them, then the sigmoid gate is applied as the
//! epilogue. A tile gets pieces in proportion to its cells, and a launch never has more
//! blocks than the device holds at once: a few blocks too many wait for a second wave that
//! runs almost alone, which cost the first version of these kernels a fifth of its time.
//!
//! There are three kernel families, chosen per launch inside `attend`:
//!
//! - **Direct** (`kernels/attn_direct.cuh`), for items of up to 8 vectors (one row of a
//!   decode step) in converted entries with K and V both quantized. No tensor cores and no
//!   staging. For the scores every lane takes the row of its own cell and the integer dot
//!   product of its codes with the queries, which are rotated and quantized to 15 bits and
//!   a scale per group once per block; the sum of a group is scaled once. For the output
//!   every lane holds 8 values of every vector and adds weight times code times scale in
//!   f32, one cell after the other. Three blocks of four warps per multiprocessor.
//!   The rows of a [`Planner::window`] never run on it: a row of a draft window must
//!   compute as the row alone does, and a window of rows has no kernel with this
//!   arithmetic that reads the cells once.
//! - **Tensor cores** (`kernels/attn_kernel.cuh`), for everything else: exact entries, an
//!   `f16` format, more than 8 vectors. K and V of a chunk of 32 cells (16 for the wide
//!   kernel) are copied raw from global memory by `cp.async`, decoded to f16 tiles in
//!   shared memory (the rotated domain for the `qB` formats; the tile mean is not part of
//!   the tile and enters the scores and the accumulator once per tile) and multiplied with
//!   f32 accumulation. A decode step has few vectors, so the cells are the rows of the
//!   product (16 of them) and the vectors the columns (tiles of 8); a wide step has the
//!   vectors as rows. The queries are rotated once per block.
//! - **Prompt** (`kernels/attn_prompt.cuh`, [`AttnShape::Prompt128`]), for the rows of a
//!   prompt: items of up to 128 vectors over one range of cells, 8 warps of 16 vectors, chunks
//!   of 32 cells, the stored rows of the next chunk prefetched into registers. These are bound
//!   by the tensor cores, so with K in `q8` to `q4` the scores are integer products (the
//!   query as 15-bit integers, high and low bytes, against the codes of K, exact in int32, then
//!   the group's scale; closer to the reference than f16 operands) and P V sums the 32 cells of
//!   a chunk in f16 before adding them to the f32 accumulator.
//!
//! **Error**, against `KvHead::attend` on the same stored bytes, relative to the largest
//! output element. The direct kernel: 2e-5 to 1e-4 (its only inexact operand is the query,
//! 15 bits per group; codes times scales and the output are exact f32). The tensor-core
//! kernels: about 1e-3 (5e-4 to 3e-3 on random rows and on the captured rows of
//! Qwen3.8-27B, 6e-3 at worst), from their f16 operands: the scaled query, the decoded K
//! and V (code times scale rounded once to f16) and the softmax weights. Neither depends on
//! the format: the formats' own errors are in the stored bytes, which the kernels read
//! exactly. A row of a policy with exact entries (`q7` to `q4`) is a merge of both.
//! Accumulating the products in f16 with a flush to f32, which doubles the rate of the
//! tensor cores on capped GeForce cards, was built and measured for decode: for the scores it
//! costs up to 3e-2, and for the output it is slower than f32 accumulation (those kernels are
//! not bound by the tensor cores), so they have no such path.

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

/// The bit of a converted entry that puts it in the far pool (page-locked host
/// memory), as `ATTN_FAR_BIT` in `kernels/attn_layout.cuh`: bits 0 to 29 are the
/// slot there.
pub const FAR_ENTRY: u32 = 1 << 30;

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
    /// The prompt kernels that compute the scores in integers, by head size.
    prompt_int: Vec<(u32, Function)>,
    /// The direct kernels by head size and the vectors their registers hold.
    direct: Vec<(u32, u32, Function)>,
    merge: Vec<(u32, Function)>,
}

/// A plan on the device.
#[derive(Debug)]
pub struct PlanBuffers {
    items: DevicePtr,
    ranges: DevicePtr,
    rows: DevicePtr,
    row_ids: DevicePtr,
    /// The buffer [`Plan::upload`] allocated; `None` when the caller owns the memory.
    _owned: Option<DeviceBuffer>,
    /// Per launch: the shape, the first item, the number of items, the most rows of an
    /// item, and whether the items may run on the direct kernel (converted entries, and
    /// none bound to the tensor cores).
    groups: Vec<(AttnShape, usize, usize, u32, bool)>,
    n_rows: u32,
    state_slots: u64,
    needs_merge: bool,
}

impl PlanBuffers {
    /// The rows of the plan on the device (for the append kernel).
    pub fn rows(&self) -> DevicePtr {
        self.rows
    }
}

/// Where the four arrays of a plan lie in one block of bytes: offsets from its start,
/// each a multiple of 16, and the length of the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanLayout {
    /// Offset of the items (48 bytes each).
    pub items: u64,
    /// Offset of the ranges of cells (three `u32` each: start, end, cells before).
    pub ranges: u64,
    /// Offset of the rows (16 bytes each).
    pub rows: u64,
    /// Offset of the row lists of the items (`u32` each).
    pub row_ids: u64,
    /// Length of the block.
    pub bytes: u64,
}

impl Plan {
    /// The layout of the plan's arrays in one block of bytes ([`Plan::write`]).
    pub fn layout(&self) -> PlanLayout {
        let pad = |n: usize| (n as u64).next_multiple_of(16);
        let items = 0;
        let ranges = items + pad(self.items.len() * 48);
        let rows = ranges + pad(self.ranges.len() * 4);
        let row_ids = rows + pad(self.rows.len() * 16);
        let bytes = (row_ids + pad(self.row_ids.len() * 4)).max(16);
        PlanLayout { items, ranges, rows, row_ids, bytes }
    }

    /// Writes the plan's arrays into `dst` as the kernels read them, at the offsets of
    /// [`Plan::layout`]. `dst` is memory of the caller, for example a part of the
    /// page-locked record of a step that goes to the device with one copy.
    ///
    /// # Panics
    ///
    /// If `dst` is shorter than `layout().bytes`.
    pub fn write(&self, dst: &mut [u8]) {
        let layout = self.layout();
        assert!(dst.len() as u64 >= layout.bytes, "the plan needs {} bytes", layout.bytes);
        let mut at = layout.items as usize;
        for item in &self.items {
            let words = [
                item.table,
                item.row0,
                item.n_rows,
                item.kv_head,
                item.ranges_off,
                item.n_ranges,
                item.total,
                item.flags,
                item.slot,
                item.swap_tile,
                item.swap_entry,
                item.pad,
            ];
            for w in words {
                dst[at..at + 4].copy_from_slice(&w.to_le_bytes());
                at += 4;
            }
        }
        let mut put = |offset: u64, words: &mut dyn Iterator<Item = u32>| {
            let mut at = offset as usize;
            for w in words {
                dst[at..at + 4].copy_from_slice(&w.to_le_bytes());
                at += 4;
            }
        };
        put(layout.ranges, &mut self.ranges.iter().copied());
        put(
            layout.rows,
            &mut self.rows.iter().flat_map(|r| [r.table, r.pos, r.slot_base, r.n_slots]),
        );
        put(layout.row_ids, &mut self.row_ids.iter().copied());
    }

    /// The plan with its arrays in device memory the caller owns: the bytes of
    /// [`Plan::write`] at `base`. Nothing is allocated or copied here, so a step loop
    /// that carries the plan in its step record pays nothing per step.
    ///
    /// # Safety
    ///
    /// When a launch that was given the returned buffers runs, `base` must address
    /// `layout().bytes` bytes of live memory on the launch's device that hold the bytes
    /// [`Plan::write`] produced for this plan.
    pub unsafe fn on_device(&self, base: DevicePtr) -> PlanBuffers {
        let layout = self.layout();
        PlanBuffers {
            items: base.offset(layout.items),
            ranges: base.offset(layout.ranges),
            rows: base.offset(layout.rows),
            row_ids: base.offset(layout.row_ids),
            _owned: None,
            groups: self
                .groups
                .iter()
                .map(|(s, r)| {
                    let items = &self.items[r.clone()];
                    let rows = items.iter().map(|i| i.n_rows).max().unwrap_or(0);
                    let direct = items.iter().all(AttnItem::direct);
                    (*s, r.start, r.end - r.start, rows, direct)
                })
                .collect(),
            n_rows: self.rows.len() as u32,
            state_slots: u64::from(self.state_slots),
            needs_merge: self.needs_merge,
        }
    }

    /// Uploads the plan into a buffer of its own (synchronously: an allocation and a
    /// copy that waits for the device). A step loop avoids both with [`Plan::write`]
    /// and [`Plan::on_device`].
    pub fn upload(&self, device: &Device) -> Result<PlanBuffers> {
        let mut bytes = vec![0u8; self.layout().bytes as usize];
        self.write(&mut bytes);
        let buffer = DeviceBuffer::new(device, bytes.len() as u64)?;
        buffer.write(0, &bytes)?;
        // SAFETY: the buffer holds the plan's bytes and lives as long as the result.
        let mut buffers = unsafe { self.on_device(buffer.ptr()) };
        buffers._owned = Some(buffer);
        Ok(buffers)
    }
}

/// The buffers an attention launch reads and writes.
#[derive(Clone, Copy, Debug)]
pub struct AttendBuffers {
    /// The u32 entry array the plan's tables index.
    pub tables: DevicePtr,
    pub exact_pool: DevicePtr,
    pub converted_pool: DevicePtr,
    /// The converted slots of entries with [`FAR_ENTRY`] (null when no entry has it).
    pub far_pool: DevicePtr,
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
        let mut prompt_int = Vec::new();
        for d in [256u32, 128] {
            for shape in AttnShape::ALL {
                let f = module.function(&shape.kernel_name(d))?;
                f.allow_dynamic_shared(shape.smem_bytes(d))?;
                attend.push((d, shape, f));
            }
            merge.push((d, module.function(&format!("ares_attn_merge_d{d}"))?));
            let f = module.function(&format!("ares_attn_d{d}_p128i"))?;
            f.allow_dynamic_shared(AttnShape::Prompt128.smem_bytes(d))?;
            prompt_int.push((d, f));
        }
        let mut direct = Vec::new();
        for d in [256u32, 128] {
            for nv in DIRECT_VECS {
                let f = module.function(&format!("ares_attn_d{d}_r{nv}"))?;
                direct.push((d, nv, f));
            }
        }
        Ok(Self {
            device: device.clone(),
            append: module.function("ares_attn_append")?,
            convert: module.function("ares_attn_convert")?,
            encode_tile_f32: module.function("ares_attn_encode_tile_f32")?,
            attend,
            prompt_int,
            direct,
            merge,
        })
    }

    /// Attention of the rows of a plan: one launch per kernel shape and kind of entry the
    /// plan uses, then the merge of partial states if any row has more than one piece.
    /// Items of up to 8 vectors in converted entries with K and V both quantized run on
    /// the direct kernel, everything else on the tensor-core kernels, and so do the
    /// items of a [`Planner::window`], whose rows must compute the same alone and
    /// together. `scale` is the
    /// factor on the scores (`1 / sqrt(head_dim)`).
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
        for (shape, first, count, max_rows, direct) in &plan.groups {
            let vecs = max_rows * geo.q_per_kv;
            let quantized = *direct && geo.k != CacheFormat::F16 && geo.v != CacheFormat::F16;
            if quantized && vecs <= DIRECT_MAX_VECS {
                let nv = DIRECT_VECS.into_iter().find(|nv| vecs <= *nv).unwrap_or(DIRECT_MAX_VECS);
                let function = self
                    .direct
                    .iter()
                    .find(|(dd, n, _)| *dd == d && *n == nv)
                    .map(|(_, _, f)| f)
                    .ok_or_else(|| Error::Invalid {
                        what: "attention",
                        detail: format!("no kernel for a head size of {d}"),
                    })?;
                let args = Args::new()
                    .ptr(plan.items.offset(*first as u64 * 48))
                    .ptr(plan.ranges)
                    .ptr(plan.rows)
                    .ptr(plan.row_ids)
                    .ptr(buffers.tables)
                    .ptr(buffers.converted_pool)
                    .ptr(buffers.far_pool)
                    .ptr(buffers.q)
                    .ptr(gate)
                    .ptr(buffers.out)
                    .ptr(st_acc)
                    .ptr(st_lse)
                    .u64(geo.config())
                    .f32(scale * std::f32::consts::LOG2_E);
                // SAFETY: the caller's contract.
                unsafe {
                    function.launch(
                        stream,
                        [*count as u32, 1, 1],
                        [32 * DIRECT_WARPS, 1, 1],
                        direct_smem_bytes(d, DIRECT_WARPS),
                        args,
                    )?;
                }
                continue;
            }
            // Converted entries with K quantized: the prompt kernel's scores in integers.
            let integer = *shape == AttnShape::Prompt128 && *direct && geo.k != CacheFormat::F16;
            let function = if integer {
                self.prompt_int.iter().find(|(dd, _)| *dd == d).map(|(_, f)| f)
            } else {
                self.attend.iter().find(|(dd, s, _)| *dd == d && s == shape).map(|(_, _, f)| f)
            }
            .ok_or_else(|| Error::Invalid {
                what: "attention",
                detail: format!("no kernel for a head size of {d}"),
            })?;
            let args = Args::new()
                .ptr(plan.items.offset(*first as u64 * 48))
                .ptr(plan.ranges)
                .ptr(plan.rows)
                .ptr(plan.row_ids)
                .ptr(buffers.tables)
                .ptr(buffers.exact_pool)
                .ptr(buffers.converted_pool)
                .ptr(buffers.far_pool)
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
                .ptr(plan.rows)
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
    /// entry array the rows' `table` fields index; `far_pool` holds the slots
    /// of entries with [`FAR_ENTRY`] (null when none has it); `status` is one
    /// u32 the kernel or-s the `ATTN_STATUS_*` bits into.
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
        far_pool: DevicePtr,
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
            .ptr(far_pool)
            .u64(geo.config())
            .ptr(status);
        // SAFETY: the caller's contract.
        unsafe { self.append.launch(stream, [n_rows, geo.kv_heads, 1], [64, 1, 1], 0, args) }
    }

    /// Converts exact slots into converted ones: `jobs` is `n_jobs` pairs of u32
    /// (exact slot, converted slot; with [`FAR_ENTRY`] a slot of `far_pool`).
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
        far_pool: DevicePtr,
        status: DevicePtr,
    ) -> Result<()> {
        geo.check()?;
        if n_jobs == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(jobs)
            .ptr(exact_pool)
            .ptr(converted_pool)
            .ptr(far_pool)
            .u64(geo.config())
            .ptr(status);
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

/// The vectors the registers of the compiled direct kernels hold.
const DIRECT_VECS: [u32; 4] = [2, 4, 6, 8];
/// The most vectors of an item the direct kernels take.
const DIRECT_MAX_VECS: u32 = 8;
/// Warps of a block of the direct kernels (`ATTN_DIRECT_WARPS` in `kernels/attention.cu`).
const DIRECT_WARPS: u32 = 4;

/// Shared memory of a block of the direct kernel: the layout of `attn_direct_body` in
/// `kernels/attn_direct.cuh`.
const fn direct_smem_bytes(head_dim: u32, warps: u32) -> u32 {
    head_dim * (32 + 16 + 2) + warps * (32 * 8 * 4 + 8 * 4 + 32 * 8 + 32 * 4) + warps * 64
}

/// Floats of partial outputs in the scratch (the log sums follow).
fn plan_state_floats(plan: &PlanBuffers, hq: u64, d: u32) -> u64 {
    plan.state_slots * hq * u64::from(d)
}
