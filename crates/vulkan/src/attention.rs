//! Attention on the device: writing K and V into the cache in the engine's own formats, and
//! attention that reads them natively.
//!
//! This is the module of `ares_cuda`'s `attention` for Vulkan: the same formats (the bytes of
//! `ares_quant::kv`, written bit for bit), the same layout of a sequence on the device, the
//! same plan (items, rows, ranges, state slots) and the same calls ([`AttentionKernels::append`],
//! [`convert`](AttentionKernels::convert), [`attend`](AttentionKernels::attend),
//! [`encode_tile_f32`](AttentionKernels::encode_tile_f32)), so that a step loop that builds a
//! plan and the pools for one backend builds them for the other. `ares_quant::kvhead` is the
//! reference both are held to, by the same tests.
//!
//! # The stored sequence on the device
//!
//! A sequence is a **table of entries**, one `u32` per [`ENTRY_CELLS`] (128) cells, which is
//! one tile of the formats with a mean. Bit 31 of an entry says which pool it is in (set: the
//! converted pool), the low 31 bits are the slot in that pool. The table is the same for every
//! attention layer; each layer and device has its own pair of pools, indexed by the slot,
//! holding only the KV heads of that device.
//!
//! - An **exact** slot holds, for each KV head, 128 rows of K and 128 rows of V as f16 (`2 * D`
//!   bytes per row, K first). [`CacheGeometry::exact_entry_bytes`].
//! - A **converted** slot holds, for each KV head, `[K mean][K rows][V mean][V rows]`: 128 rows
//!   in the format of the policy (`format.row_bytes(D)` each, the bytes `KvHead` stores) and,
//!   for q7 to q4, the mean of the tile's rotated rows as `D` f16 before them.
//!   [`CacheGeometry::converted_entry_bytes`]. All sections start at a multiple of 16 bytes.
//!
//! With a mean in K or V (q7 to q4), the first tile and the newest tiles of a sequence are exact
//! entries, and a tile becomes a converted entry ([`AttentionKernels::convert`]) when the policy
//! says so. Without one (f16 and q8), cells are written straight into the format in converted
//! entries, the entry that is still filling included. A draft row of such a policy is encoded
//! from its f32 values like any other, whereas the reference keeps a pending cell as f16 until
//! it is settled; the stored bytes of an accepted draft then differ by that one f16 rounding
//! of the input, and rejected drafts are forgotten by the table shrinking.
//!
//! What `ares-state` has to provide is the same as for CUDA (a page is one entry, the map from
//! page to slot is per page, the state manager owns two slot allocators, conversions are
//! decided by `KvPolicy::quantized_cells`): see `ares_cuda::AttentionKernels`.
//!
//! # Reading
//!
//! [`Planner`] cuts a step into **items**, one workgroup each, and
//! [`AttentionKernels::attend`] launches one kernel per shape. An item is some rows (their query
//! heads of one KV head are the "vectors" of the product) and a list of ranges of cells, in
//! entries of one kind. Rows of a decode window share the K and V they read, and so do the rows
//! of several sequences over a prefix they have in common; dense causal attention is one range
//! per item, block-sparse attention (qwen4exp) one range per selected block. Items that are
//! only part of a row's cells write a state (the log2 of the sum of exponentials, the weighted
//! V in the original domain) and [`attend`](AttentionKernels::attend) merges them, then the
//! sigmoid gate is applied as the epilogue. A tile gets pieces in proportion to its cells, and
//! a launch never has many more workgroups than the device holds at once
//! ([`AttentionKernels::plan_config`]).
//!
//! # Against `ares_cuda`
//!
//! - **One kernel family, no tensor cores.** An item holds up to 16 vectors (8 on a device with
//!   less than 28 KiB of shared memory) where CUDA's go up to 64 with tensor cores. The kernel
//!   (`shaders/attn_item.glsl`) takes one cell per invocation for the scores and a slice of the
//!   columns per invocation for V, in f32 throughout, with the stored rows decoded in registers.
//!   Its error against the reference is therefore f32 rounding (1e-5 of the largest output
//!   element at worst on random rows, the captured rows of Qwen3.8-27B and every pair of
//!   formats, mostly 1e-6; `ares_quant::kvhead::OUTPUT_TOLERANCE` is 2e-5), where CUDA's
//!   tensor-core kernels have about 1e-3 from their f16 operands. A
//!   window of more than 2 rows of 6 query heads is several items that each read the same
//!   cells (see [`rows_per_tile`](PlanConfig)), where CUDA reads them once with 64 vectors.
//! - Head sizes 128 and 256, as CUDA's kernels; at most [`AttentionKernels::max_vectors`] query
//!   heads per KV head.
//! - Pipelines are made when a (head size, shape, K format, V format) is first launched, or
//!   by [`AttentionKernels::prepare`]. The formats of K and V are specialization constants of
//!   the pipeline.
//! - The device needs what the quantized weight kernels need and shuffles within a subgroup;
//!   subgroups of 32 or a multiple of it ([`AttentionKernels::load`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::{grid_1d, parallel_map};
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::module::{Args, Function, Lazy, Module};
use crate::stream::{Event, Stream};
use crate::weights::image;

mod plan;

pub use plan::{AttnItem, AttnShape, Plan, PlanConfig, Planner};

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
        let bad = |detail: String| Err(Error::invalid("cache geometry", detail));
        if self.head_dim != 128 && self.head_dim != 256 {
            return bad(format!("head size {} is not 128 or 256", self.head_dim));
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
pub struct AttentionKernels(Arc<Set>);

#[derive(Debug)]
struct Set {
    device: Device,
    append: Arc<Lazy>,
    convert: Arc<Lazy>,
    encode_tile_f32: Arc<Lazy>,
    merge: Arc<Lazy>,
    /// The attention kernels made so far, by (head size, shape, K format, V format).
    items: Mutex<HashMap<(u32, AttnShape, u32, u32), Function>>,
    /// The largest shape the device's shared memory holds.
    max_shape: AttnShape,
    /// The workgroups a launch aims for: three for each of the device's multiprocessors.
    resident: u32,
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
    /// Per launch: the shape, the first item and the number of items.
    groups: Vec<(AttnShape, usize, usize)>,
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
                item.pad[0],
                item.pad[1],
                item.pad[2],
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
            groups: self.groups.iter().map(|(s, r)| (*s, r.start, r.end - r.start)).collect(),
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
    /// `[row][q_heads_of_model][head_dim]` f32, scaled by nothing (the kernel applies `scale`).
    pub q: DevicePtr,
    /// The gate (same layout as `q`), applied as `out * sigmoid(gate)` when present.
    pub gate: Option<DevicePtr>,
    /// `[row][q_heads_of_model][head_dim]` f32; the rows and heads of this plan are written.
    pub out: DevicePtr,
    /// Scratch for partial states: [`Plan::state_bytes`] bytes.
    pub scratch: DevicePtr,
}

/// Floats of partial outputs in the scratch (the log sums follow).
fn plan_state_floats(plan: &PlanBuffers, hq: u64, d: u32) -> u64 {
    plan.state_slots * hq * u64::from(d)
}

impl AttentionKernels {
    /// Prepares the module on a device. No attention pipeline is made yet (one is made when a
    /// head size, shape and pair of formats is first launched, or by
    /// [`prepare`](Self::prepare)); the writing kernels are made on first use too.
    ///
    /// Refused where the kernels cannot run: a device without what the quantized weight kernels
    /// need (integer dot products, 8- and 16-bit buffer access, clustered subgroup operations,
    /// subgroups that are a multiple of 32 wide), without shuffles within a subgroup, or with
    /// too little shared memory for the smallest kernel. The size of the device (how many large
    /// workgroups it runs at once) is measured here (about 0.2 s): [`plan_config`](Self::plan_config).
    pub fn load(device: &Device) -> Result<Self> {
        if let Some(reason) = &device.physical().attention_unsupported {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: format!(
                    "the attention kernels need what the weight kernels need: {reason}"
                ),
            });
        }
        let shared = device.physical().max_shared_memory;
        let max_shape = AttnShape::ALL
            .into_iter()
            .rev()
            .find(|s| s.shared_bytes(256) <= shared)
            .ok_or_else(|| Error::Unsupported {
                device: device.info().name.clone(),
                reason: format!(
                    "the attention kernels need {} bytes of shared memory, the device has {shared}",
                    AttnShape::Vec8.shared_bytes(256)
                ),
            })?;
        let lazy = |name: &str| -> Result<Arc<Lazy>> { Ok(Lazy::new(device, image(name)?)) };
        let resident = 3 * resident_workgroups(device)?;
        Ok(Self(Arc::new(Set {
            device: device.clone(),
            append: lazy("attn_append")?,
            convert: lazy("attn_convert")?,
            encode_tile_f32: lazy("attn_encode_tile_f32")?,
            merge: lazy("attn_merge")?,
            items: Mutex::new(HashMap::new()),
            max_shape,
            resident,
        })))
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// The most vectors (query heads of the rows) an item holds on this device: 16, or 8 where
    /// the shared memory is small. A model needs `q_per_kv` of them at least.
    pub fn max_vectors(&self) -> u32 {
        self.0.max_shape.max_vecs()
    }

    /// The configuration of the planner for this device: three workgroups for each of its
    /// multiprocessors (the size of the device is measured at load, see `resident_workgroups`) as
    /// the number a decode step is cut into, and the vectors an item holds.
    pub fn plan_config(&self) -> PlanConfig {
        PlanConfig::new(self.0.resident, self.max_vectors())
    }

    /// Makes the pipelines the geometry needs now, on all cores: the attention kernels of
    /// its head size and K and V formats, the writing kernels and the merge. The runtime calls
    /// this when it loads a model, so that no step pays for a compilation.
    pub fn prepare(&self, geo: &CacheGeometry) -> Result<()> {
        geo.check()?;
        let shapes: Vec<AttnShape> =
            AttnShape::ALL.into_iter().filter(|s| *s <= self.0.max_shape).collect();
        parallel_map(&shapes, |shape| self.item_function(geo, *shape).map(|_| ()))?;
        let lazies = [&self.0.append, &self.0.convert, &self.0.merge];
        parallel_map(&lazies, |lazy| lazy.get().map(|_| ())).map(|_| ())
    }

    /// The attention pipeline for a head size, a shape and the formats of K and V.
    fn item_function(&self, geo: &CacheGeometry, shape: AttnShape) -> Result<Function> {
        let key = (geo.head_dim, shape, geo.k.id(), geo.v.id());
        if let Some(f) =
            self.0.items.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&key)
        {
            return Ok(f.clone());
        }
        if shape > self.0.max_shape {
            return Err(Error::invalid(
                "attention",
                format!(
                    "an item of up to {} vectors needs {} bytes of shared memory, the device has {}",
                    shape.max_vecs(),
                    shape.shared_bytes(geo.head_dim),
                    self.0.device.physical().max_shared_memory
                ),
            ));
        }
        let name = shape.kernel_name(geo.head_dim);
        let made = Module::load(&self.0.device, image(&name)?)?
            .function_with("main", &[(0, geo.k.id()), (1, geo.v.id())])?;
        let mut items = self.0.items.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(items.entry(key).or_insert(made).clone())
    }

    /// Attention of the rows of a plan: one launch per kernel shape the plan uses, then the
    /// merge of partial states if any row has more than one piece. `scale` is the factor on the
    /// scores (`1 / sqrt(head_dim)`).
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
        if geo.q_per_kv > self.max_vectors() {
            return Err(Error::invalid(
                "attention",
                format!(
                    "{} query heads per KV head; an item holds {} vectors on this device",
                    geo.q_per_kv,
                    self.max_vectors()
                ),
            ));
        }
        let d = geo.head_dim;
        let hq = u64::from(geo.kv_heads * geo.q_per_kv);
        let acc_bytes = plan_state_floats(plan, hq, d) * 4;
        let st_acc = buffers.scratch;
        let st_lse = buffers.scratch.offset(acc_bytes);
        let gate = buffers.gate.unwrap_or(DevicePtr::NULL);
        let limit = self.0.device.physical().max_work_group_count;
        for (shape, first, count) in &plan.groups {
            let function = self.item_function(geo, *shape)?;
            let args = Args::new()
                .ptr(plan.items.offset(*first as u64 * 48))
                .ptr(plan.ranges)
                .ptr(plan.rows)
                .ptr(plan.row_ids)
                .ptr(buffers.tables)
                .ptr(buffers.exact_pool)
                .ptr(buffers.converted_pool)
                .ptr(buffers.q)
                .ptr(gate)
                .ptr(buffers.out)
                .ptr(st_acc)
                .ptr(st_lse)
                .u64(geo.config())
                .f32(scale * std::f32::consts::LOG2_E);
            // SAFETY: the caller's contract.
            unsafe { function.launch(stream, grid_1d(*count as u32, limit[0]), args)? };
        }
        if plan.needs_merge {
            let function = self.0.merge.get()?;
            let args = Args::new()
                .ptr(plan.rows)
                .ptr(st_acc)
                .ptr(st_lse)
                .ptr(gate)
                .ptr(buffers.out)
                .u64(geo.config());
            let grid = [plan.n_rows, geo.kv_heads * geo.q_per_kv, 1];
            if grid[0] > limit[0] || grid[1] > limit[1] {
                return Err(Error::invalid(
                    "attention",
                    format!("a merge of {grid:?} workgroups; the device allows {limit:?}"),
                ));
            }
            // SAFETY: the caller's contract.
            unsafe { function.launch(stream, grid, args)? };
        }
        Ok(())
    }

    /// Appends the K and V rows of a step. `rows[i]` says where row `i` goes; `k` and `v` are
    /// `[row][kv_total][head_dim]` f32; `tables` is the u32 entry array the rows' `table` fields
    /// index; `status` is one u32 the kernel or-s the `ATTN_STATUS_*` bits into.
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
        let grid = [n_rows, geo.kv_heads, 1];
        self.launch_checked("append", self.0.append.get()?, grid, args, stream)
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
        let grid = [n_jobs, geo.kv_heads, 2];
        self.launch_checked("convert", self.0.convert.get()?, grid, args, stream)
    }

    /// The codec alone on f32 rows, for tests: `rows` rows of `head_dim` values (at most 4096) as
    /// one tile (`use_mean`) or each on its own. Writes the mean (`format.mean_bytes`) and the
    /// rows.
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
        if head_dim == 0 || !head_dim.is_multiple_of(32) || head_dim > 4096 {
            return Err(Error::invalid(
                "encode",
                format!("a head size of {head_dim}: a multiple of 32, at most 4096"),
            ));
        }
        let args = Args::new()
            .ptr(src)
            .ptr(mean_out)
            .ptr(rows_out)
            .ptr(status)
            .u32(rows)
            .u32(head_dim)
            .u32(format.id())
            .u32(u32::from(use_mean));
        // SAFETY: the caller's contract.
        unsafe { self.0.encode_tile_f32.get()?.launch(stream, [1, 1, 1], args) }
    }

    fn launch_checked(
        &self,
        what: &'static str,
        function: &Function,
        grid: [u32; 3],
        args: Args,
        stream: &Stream,
    ) -> Result<()> {
        let limit = self.0.device.physical().max_work_group_count;
        if grid.iter().zip(limit).any(|(g, l)| *g > l) {
            return Err(Error::invalid(
                what,
                format!("a grid of {grid:?} workgroups; the device allows {limit:?}"),
            ));
        }
        // SAFETY: the callers of this function are the unsafe launches above, which pass on
        // their caller's guarantee.
        unsafe { function.launch(stream, grid, args) }
    }
}

/// How many of the largest workgroups (up to 1024 invocations) the device runs at once, measured:
/// a kernel that spins on dependent arithmetic is launched with a growing number of workgroups;
/// while they all fit, the launch takes as long as one, and past that it takes a second wave. A
/// multiprocessor holds one or two such workgroups, whatever the registers and the shared memory
/// of a kernel, so the number follows the count of multiprocessors (or compute units), which
/// Vulkan does not report. The planner aims for three times that, as the CUDA planner does for
/// each multiprocessor ([`PlanConfig::new`]).
///
/// The clocks of a card that was idle are low for the first moments, so the measurement starts
/// after 100 ms of work, takes the best of seven launches at each number, and compares with the
/// best time of the whole series; the number is the one before two in a row that take half as long
/// again. (A first version spun with the shared memory of the attention kernel, to find how many of
/// those sit on a multiprocessor next to its registers: the driver gave two or three per
/// multiprocessor, 92 or 138 on the RTX 4070, in runs a minute apart, and the plan, and with it
/// the speed of the kernels, followed by up to 45 %.)
fn resident_workgroups(device: &Device) -> Result<u32> {
    let size = device.physical().max_workgroup_size.min(1024) / 32 * 32;
    let probe = Module::load(device, image("attn_probe")?)?.function_with("main", &[(0, size)])?;
    let out = DeviceBuffer::new(device, 4 * 4096)?;
    let stream = Stream::new(device)?;
    let (start, stop) = (Event::timed(device)?, Event::timed(device)?);
    let limit = device.physical().max_work_group_count[0].min(1024);
    let time = |groups: u32| -> Result<f32> {
        let mut best = f32::MAX;
        for _ in 0..7 {
            start.record(&stream)?;
            // SAFETY: the kernel writes one float per workgroup into `out`, which holds 4096.
            unsafe {
                probe.launch(&stream, [groups, 1, 1], Args::new().ptr(out.ptr()).u32(8192))?;
            }
            stop.record(&stream)?;
            stop.synchronize()?;
            best = best.min(stop.elapsed_ms_since(&start)?);
        }
        Ok(best)
    };
    let warm = std::time::Instant::now();
    while warm.elapsed() < std::time::Duration::from_millis(100) {
        time(16)?;
    }
    let mut ladder = vec![1u32];
    while ladder.last().is_some_and(|n| *n < limit) {
        let n = *ladder.last().unwrap_or(&1);
        ladder.push((n + (n / 8).max(1)).min(limit));
    }
    let times = ladder.iter().map(|n| time(*n)).collect::<Result<Vec<f32>>>()?;
    let base = times.iter().copied().fold(f32::MAX, f32::min);
    let slow = |i: usize| times[i] > 1.5 * base;
    let first_slow = (0..times.len().saturating_sub(1)).find(|i| slow(*i) && slow(i + 1));
    Ok(match first_slow {
        Some(i) => ladder[i.saturating_sub(1)],
        None => *ladder.last().unwrap_or(&1),
    })
}
